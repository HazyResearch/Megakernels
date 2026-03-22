from __future__ import annotations

import itertools
import math
from typing import Dict, List, Tuple

from .dag import DType, Node, OpType, TensorMeta
from .instruction import (
    IType,
    Instruction,
    OpMeta,
    MAX_DST_BARRIERS,
    MAX_DST_TENSORS,
    MAX_INDICES,
    MAX_SRC_BARRIERS,
    MAX_SRC_TENSORS,
)


MAX_BARRIERS = 256
MAX_TENSOR_ALLOCATIONS = 256

# Auto-built from all IType subclasses in instruction.py
_OPTYPE_TO_ITYPE: Dict[str, IType] = {
    cls().op_type: cls() for cls in IType.__subclasses__()
}


def _resolve_itype(node: Node) -> IType:
    if node.optype.value not in _OPTYPE_TO_ITYPE:
        raise RuntimeError(f"[MegaKittens] No IType mapping for OpType {node.optype.value}")
    return _OPTYPE_TO_ITYPE[node.optype.value]


def schedule(
    dag_nodes: List[Node],
) -> Tuple[List[OpMeta], List[TensorMeta], List[Instruction], int, Tuple[int, ...], Tuple[int, ...]]:
    """
    Convert a validated DAG into a minimal set of tensors and a flat per-SM instruction list.
    """
    # Phase 1: Tensor metadata collection
    tensor_metas: List[TensorMeta] = []
    tensor_index: Dict[Tuple[int, int], int] = {}
    input_tensor_indices: List[int] = []
    output_tensor_indices: List[int] = []

    # TODO: reuse allocated tensors
    for node in dag_nodes:
        for out_idx, tensor_meta in enumerate(node.out_tensors):
            if len(tensor_metas) >= MAX_TENSOR_ALLOCATIONS:
                raise RuntimeError(
                    f"[MegaKittens] The given compute graph requires tensor count exceeding {MAX_TENSOR_ALLOCATIONS}."
                )
            tensor_index[(id(node), out_idx)] = len(tensor_metas)
            tensor_metas.append(tensor_meta)
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
        itype = _resolve_itype(node)
        node_inst_count[id(node)] = 1
        for dim in out_shape:
            node_inst_count[id(node)] *= math.ceil(dim / itype.tile_size)

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
    opcode_counter = 0
    op_metas: List[OpMeta] = []
    opcode_map: Dict[Tuple[IType, Tuple[int, ...], Tuple[int, ...]], int] = {}

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

        src_metas = tuple(in_node.out_tensors[slot_idx] for in_node, slot_idx in node.in_nodes)
        dst_metas = node.out_tensors
        itype.validate(src_metas, dst_metas)

        key = (itype, src_tensors, dst_tensors)
        if key not in opcode_map:
            opcode = opcode_counter
            opcode_counter += 1
            opcode_map[key] = opcode
            op_metas.append(OpMeta(opcode=opcode, itype=itype, src_tensors=src_tensors, dst_tensors=dst_tensors))
        else:
            opcode = opcode_map[key]

        src_bar_list = node_src_barriers.get(id(node), [])
        src_barriers = tuple(bid for bid, _ in src_bar_list)
        src_barrier_targets = tuple(tgt for _, tgt in src_bar_list)
        dst_barrier = tuple(node_dst_barriers.get(id(node), []))

        out_shape = node.out_tensors[0].shape
        ranges = [range(math.ceil(dim / itype.tile_size)) for dim in out_shape]
        tiles = list(itertools.product(*ranges))

        for tile in tiles:
            instructions.append(Instruction(
                opcode=opcode,
                src_tensors=src_tensors,
                dst_tensors=dst_tensors,
                indices=tile,
                src_barriers=src_barriers,
                src_barrier_targets=src_barrier_targets,
                dst_barrier=dst_barrier,
            ))

    return (
        op_metas,
        tensor_metas,
        instructions,
        barrier_counter,
        tuple(input_tensor_indices),
        tuple(output_tensor_indices),
    )
