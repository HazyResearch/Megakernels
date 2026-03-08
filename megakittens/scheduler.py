from __future__ import annotations

import itertools
import math
from typing import Dict, List, Tuple

# TODO: completely remove torch dependency in scheduler and direcly rely on CUDA API
import torch

from .dag import DType, Node, OpType
from .instruction import (
    IType,
    Instruction,
    MAX_DST_BARRIERS,
    MAX_DST_TENSORS,
    MAX_INDICES,
    MAX_SRC_BARRIERS,
    MAX_SRC_TENSORS,
)


MAX_BARRIERS = 256
MAX_TENSOR_ALLOCATIONS = 256

# TODO: change
TILE_SIZE = 4

_MK_DTYPE_TO_TORCH_DTYPE: Dict[DType, torch.dtype] = {
    DType.fp64: torch.float64,
    DType.fp32: torch.float32,
    DType.bf16: torch.bfloat16,
    DType.half: torch.float16,
    DType.fp8e4m3: torch.float8_e4m3fn,
    DType.fp8e5m2: torch.float8_e5m2fnuz,
    DType.fp8e8m0: torch.float8_e8m0fnu,
    DType.fp4e2m1x2: torch.float4_e2m1fn_x2,
    DType.int64: torch.int64,
    DType.int32: torch.int32,
    DType.int16: torch.int16,
    DType.int8: torch.int8,
}

_OPTYPE_TO_ITYPE: Dict[OpType, IType] = {
    OpType.matmul: IType.mm_bf16_bf16_fp32_bf16,
    OpType.add: IType.add,
    OpType.relu: IType.relu,
}


def _resolve_itype(node: Node) -> IType:
    if node.optype == OpType.matmul:
        # TODO: support many types of matmuls
        for in_node, slot_idx in node.in_nodes:
            in_dtype = in_node.out_tensors[slot_idx].dtype
            if in_dtype != DType.bf16:
                raise RuntimeError(
                    f"[MegaKittens] Matmul requires bf16 inputs, got {in_dtype.value}"
                )
    if node.optype not in _OPTYPE_TO_ITYPE:
        raise RuntimeError(f"[MegaKittens] No IType mapping for OpType {node.optype.value}")
    return _OPTYPE_TO_ITYPE[node.optype]


def schedule(
    dag_nodes: List[Node],
) -> Tuple[List[torch.Tensor], List[Instruction], Tuple[int, ...], Tuple[int, ...]]:
    """
    Convert a validated DAG into allocated tensors and a flat per-SM instruction list.
    """
    # Phase 1: Tensor allocation
    tensors: List[torch.Tensor] = []
    tensor_index: Dict[Tuple[int, int], int] = {}
    input_tensor_indices: List[int] = []
    output_tensor_indices: List[int] = []

    # TODO: reuse allocated tensors
    for node in dag_nodes:
        for out_idx, tensor_meta in enumerate(node.out_tensors):
            if len(tensors) >= MAX_TENSOR_ALLOCATIONS:
                raise RuntimeError(
                    f"[MegaKittens] The given compute graph requires tensor count exceeding {MAX_TENSOR_ALLOCATIONS}."
                )
            torch_dtype = _MK_DTYPE_TO_TORCH_DTYPE[tensor_meta.dtype]
            device_str = str(tensor_meta.device)
            tensor_index[(id(node), out_idx)] = len(tensors)
            tensors.append(torch.empty(tensor_meta.shape, dtype=torch_dtype, device=device_str))
        if node.optype == OpType.input:
            if len(node.out_tensors) != 1:
                raise RuntimeError(
                    f"[MegaKittens] Input node has {len(node.out_tensors)} outputs (expected 1)"
                )
            input_tensor_indices.append(tensor_index[(id(node), 0)])
        elif node.optype == OpType.output:
            if len(output_tensor_indices) != 0:
                raise RuntimeError("[MegaKittens] Expected 1 output node")
            output_tensor_indices.extend(
                tensor_index[(id(in_node), slot_idx)] for in_node, slot_idx in node.in_nodes
            )
    if not input_tensor_indices:
        raise RuntimeError("[MegaKittens] Graph has no input tensors")
    if not output_tensor_indices:
        raise RuntimeError("[MegaKittens] Graph has no output tensors")

    # Phase 2: Instruction count per node
    # TODO: make op-specific
    node_inst_count: Dict[int, int] = {}
    for node in dag_nodes:
        if node.optype in (OpType.input, OpType.output):
            continue
        elif node.optype not in _OPTYPE_TO_ITYPE:
            raise RuntimeError(
                f"[MegaKittens] Unsupported OpType {node.optype} for instruction counting"
            )
        if len(node.in_nodes) > MAX_SRC_TENSORS:
            raise RuntimeError(
                f"[MegaKittens] Node has {len(node.in_nodes)} src tensors (max {MAX_SRC_TENSORS})"
            )
        out_shape = node.out_tensors[0].shape # TODO: support multi-output ops
        if node.optype == OpType.matmul: # TODO: support multi-dimensional matmuls
            assert len(out_shape) == 2, (
                f"[MegaKittens] Matmul output must be 2D, got {len(out_shape)}D"
            )
        if len(node.out_tensors) > MAX_DST_TENSORS:
            raise RuntimeError(
                f"[MegaKittens] Node has {len(node.out_tensors)} dst tensors (max {MAX_DST_TENSORS})"
            )
        node_inst_count[id(node)] = 1
        for dim in out_shape:
            node_inst_count[id(node)] *= math.ceil(dim / TILE_SIZE)

    # Phase 3: Barrier assignment
    barrier_counter = 0
    node_dst_barriers: Dict[int, List[int]] = {}
    node_src_barriers: Dict[int, List[Tuple[int, int]]] = {}

    # TODO: reuse barriers
    for node in dag_nodes:
        if node.optype in (OpType.input, OpType.output):
            continue
        elif node.optype not in _OPTYPE_TO_ITYPE:
            raise RuntimeError(
                f"[MegaKittens] Unsupported OpType {node.optype} for barrier assignment"
            )
        for in_node, _slot_idx in node.in_nodes:
            if in_node.optype in (OpType.input, OpType.output):
                continue
            elif in_node.optype not in _OPTYPE_TO_ITYPE:
                raise RuntimeError(
                    f"[MegaKittens] Unsupported input OpType {in_node.optype} for barrier assignment"
                )
            if barrier_counter >= MAX_BARRIERS:
                raise RuntimeError(
                    f"[MegaKittens] Barrier count exceeds {MAX_BARRIERS}"
                )
            bid = barrier_counter
            barrier_counter += 1
            node_dst_barriers.setdefault(id(in_node), []).append(bid)
            target = node_inst_count[id(in_node)]
            node_src_barriers.setdefault(id(node), []).append((bid, target))

    for nid, barriers in node_dst_barriers.items():
        if len(barriers) > MAX_DST_BARRIERS:
            raise RuntimeError(
                f"[MegaKittens] Node has {len(barriers)} dst barriers (max {MAX_DST_BARRIERS})"
            )
    for nid, barriers in node_src_barriers.items():
        if len(barriers) > MAX_SRC_BARRIERS:
            raise RuntimeError(
                f"[MegaKittens] Node has {len(barriers)} src barriers (max {MAX_SRC_BARRIERS})"
            )

    # Phase 4: Instruction generation
    instructions: List[Instruction] = []

    for node in dag_nodes:
        if node.optype in (OpType.input, OpType.output):
            continue
        elif node.optype not in _OPTYPE_TO_ITYPE:
            raise RuntimeError(
                f"[MegaKittens] Unsupported OpType {node.optype} during instruction generation"
            )

        itype = _resolve_itype(node)

        src_tensors = tuple(
            tensor_index[(id(in_node), slot_idx)]
            for in_node, slot_idx in node.in_nodes
        )

        dst_tensors = tuple(
            tensor_index[(id(node), slot)]
            for slot in range(len(node.out_tensors))
        )

        src_bar_list = node_src_barriers.get(id(node), [])
        src_barriers = tuple(bid for bid, _ in src_bar_list)
        src_barrier_targets = tuple(tgt for _, tgt in src_bar_list)
        dst_barrier = tuple(node_dst_barriers.get(id(node), []))

        # TODO: make op-specific
        out_shape = node.out_tensors[0].shape
        ranges = [range(math.ceil(dim / TILE_SIZE)) for dim in out_shape]
        tiles = list(itertools.product(*ranges))

        for tile in tiles:
            instructions.append(Instruction(
                itype=itype,
                src_tensors=src_tensors,
                dst_tensors=dst_tensors,
                indices=tile,
                src_barriers=src_barriers,
                src_barrier_targets=src_barrier_targets,
                dst_barrier=dst_barrier,
            ))

    return (
        tensors,
        instructions,
        tuple(input_tensor_indices),
        tuple(output_tensor_indices),
    )
