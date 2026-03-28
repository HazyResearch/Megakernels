from __future__ import annotations

from typing import Dict, List, Tuple

from .dispatcher import Dispatcher
from .itypes.noop import Noop
from .jit.cuda_utils import get_sm_count
from .schema.dag import DAG, OpType
from .schema.tensor import TensorMeta
from .schema.instruction import (
    IType,
    Instruction,
    InstructionMeta,
)


# Derived from the instruction struct
MAX_BARRIERS = 2**32 - 1
MAX_TENSOR_ALLOCATIONS = 256


def schedule(
    dag: DAG,
) -> Tuple[List[InstructionMeta], List[TensorMeta], List[Instruction], int, Tuple[int, ...], Tuple[int, ...]]:
    """
    Convert a validated DAG into a minimal set of tensors and a flat per-SM instruction list.
    """
    # Phase 1: Instruction count per node
    node_inst_count: Dict[int, int] = {}
    node_inst_offset: Dict[int, int] = {}
    cumulative_offset = 0
    for node in dag.nodes:
        if node.optype in (OpType.input, OpType.output):
            continue
        if len(node.in_nodes) > Instruction.MAX_SRC_TENSORS:
            raise RuntimeError(
                f"[MegaKittens] Node has {len(node.in_nodes)} src tensors (max {Instruction.MAX_SRC_TENSORS})"
            )
        if len(node.out_tensors) > Instruction.MAX_DST_TENSORS:
            raise RuntimeError(
                f"[MegaKittens] Node has {len(node.out_tensors)} dst tensors (max {Instruction.MAX_DST_TENSORS})"
            )
        itype = IType.from_optype(node.optype.value)
        src_metas = tuple(in_node.out_tensors[slot_idx] for in_node, slot_idx in node.in_nodes)
        count = itype.num_instructions(src_metas, node.out_tensors)
        node_inst_count[node.id] = count
        node_inst_offset[node.id] = cumulative_offset
        cumulative_offset += count + (-count) % Dispatcher.CLUSTER_SIZE

    # Phase 2: Tensor metadata collection
    tensor_metas: List[TensorMeta] = []
    tensor_index: Dict[Tuple[int, int], int] = {}
    input_tensor_indices: List[int] = []
    output_tensor_indices: List[int] = []
    tensor_pool: Dict[TensorMeta, List[Tuple[List[int], int, int, int]]] = {}
    release_barriers: List[Tuple[List[int], int, int]] = []

    for node in dag.nodes:
        for out_idx, tensor_meta in enumerate(node.out_tensors):
            reused = False
            if node.optype != OpType.input and node.optype != OpType.output:
                free_tensors = tensor_pool.get(tensor_meta, [])
                for i, (consumer_node_ids, last_consumer_inst, num_consumer_insts, tid) in enumerate(free_tensors):
                    if node_inst_offset[node.id] - last_consumer_inst >= get_sm_count():
                        tensor_index[(node.id, out_idx)] = tid
                        free_tensors.pop(i)
                        release_barriers.append((consumer_node_ids, num_consumer_insts, node.id))
                        reused = True
                        break

            if not reused:
                if len(tensor_metas) >= MAX_TENSOR_ALLOCATIONS:
                    raise RuntimeError(
                        f"[MegaKittens] The given compute graph requires tensor count exceeding {MAX_TENSOR_ALLOCATIONS}."
                    )
                tensor_index[(node.id, out_idx)] = len(tensor_metas)
                tensor_metas.append(tensor_meta)

            consumer_nodes = [node] + node.out_nodes[out_idx]
            if node.optype == OpType.input or any(c.optype == OpType.output for c in consumer_nodes):
                continue
            consumer_node_ids = [c.id for c in consumer_nodes]
            num_consumer_insts = sum(node_inst_count[cid] for cid in consumer_node_ids)
            last_consumer_inst = max(node_inst_offset[c.id] + node_inst_count[c.id] + (-node_inst_count[c.id]) % Dispatcher.CLUSTER_SIZE for c in consumer_nodes)
            tensor_pool.setdefault(tensor_meta, []).append((consumer_node_ids, last_consumer_inst, num_consumer_insts, tensor_index[(node.id, out_idx)]))

        if node.optype == OpType.input:
            if len(node.out_tensors) != 1:
                raise RuntimeError(
                    f"[MegaKittens] Input node has {len(node.out_tensors)} outputs (expected 1)"
                )
            input_tensor_indices.append(tensor_index[(node.id, 0)])
        elif node.optype == OpType.output:
            if len(output_tensor_indices) != 0:
                raise RuntimeError("[MegaKittens] Expected 1 output node")
            output_tensor_indices.extend(
                tensor_index[(in_node.id, slot_idx)] for in_node, slot_idx in node.in_nodes
            )
    if not input_tensor_indices:
        raise RuntimeError("[MegaKittens] Graph has no input tensors")
    if not output_tensor_indices:
        raise RuntimeError("[MegaKittens] Graph has no output tensors")

    # Phase 3: Barrier assignment
    barrier_counter = 0
    node_dst_barriers: Dict[int, List[int]] = {}
    node_src_barriers: Dict[int, List[Tuple[int, int]]] = {}
    node_num_input_barriers: Dict[int, int] = {}
    node_num_reuse_barriers: Dict[int, int] = {}

    # TODO: reuse barriers
    # Input dependency barriers
    for node in dag.nodes:
        if node.optype in (OpType.input, OpType.output):
            continue
        for in_node, _slot_idx in node.in_nodes:
            if in_node.optype in (OpType.input, OpType.output):
                continue
            if barrier_counter >= MAX_BARRIERS:
                raise RuntimeError(f"[MegaKittens] Barrier count exceeds {MAX_BARRIERS}")
            bid = barrier_counter
            barrier_counter += 1
            node_dst_barriers.setdefault(in_node.id, []).append(bid)
            target = node_inst_count[in_node.id]
            node_src_barriers.setdefault(node.id, []).append((bid, target))
            node_num_input_barriers[node.id] = node_num_input_barriers.get(node.id, 0) + 1

    # Tensor reuse barriers
    for consumer_node_ids, num_consumer_insts, reuser_id in release_barriers:
        if barrier_counter >= MAX_BARRIERS:
            raise RuntimeError(f"[MegaKittens] Barrier count exceeds {MAX_BARRIERS}")
        bid = barrier_counter
        barrier_counter += 1
        for nid in consumer_node_ids:
            node_dst_barriers.setdefault(nid, []).append(bid)
        node_src_barriers.setdefault(reuser_id, []).append((bid, num_consumer_insts))
        node_num_reuse_barriers[reuser_id] = node_num_reuse_barriers.get(reuser_id, 0) + 1

    for nid, barriers in node_dst_barriers.items():
        if len(barriers) > Instruction.MAX_DST_BARRIERS:
            raise RuntimeError(
                f"[MegaKittens] Node has {len(barriers)} dst barriers (max {Instruction.MAX_DST_BARRIERS})"
            )
    for nid, barriers in node_src_barriers.items():
        if len(barriers) > Instruction.MAX_SRC_BARRIERS:
            raise RuntimeError(
                f"[MegaKittens] Node has {len(barriers)} src barriers (max {Instruction.MAX_SRC_BARRIERS})"
            )

    # Phase 4: Instruction generation
    instructions: List[Instruction] = []
    icode_counter = 1  # icode 0 is reserved for noop
    instruction_metas: List[InstructionMeta] = [InstructionMeta(icode=0, itype=Noop(), src_tensors=(), dst_tensors=())]
    icode_map: Dict[Tuple[IType, Tuple[int, ...], Tuple[int, ...]], int] = {}
    noop = Instruction(icode=0, src_tensors=(), dst_tensors=(), indices=(), src_barriers=(), src_barrier_targets=(), num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0, dst_barriers=())

    for node in dag.nodes:
        if node.optype in (OpType.input, OpType.output):
            continue

        itype = IType.from_optype(node.optype.value)

        src_tensors = tuple(
            tensor_index[(in_node.id, slot_idx)]
            for in_node, slot_idx in node.in_nodes
        )

        dst_tensors = tuple(
            tensor_index[(node.id, slot)]
            for slot in range(len(node.out_tensors))
        )

        src_metas = tuple(in_node.out_tensors[slot_idx] for in_node, slot_idx in node.in_nodes)
        dst_metas = node.out_tensors
        itype.validate(src_metas, dst_metas)

        key = (itype, src_tensors, dst_tensors)
        if key not in icode_map:
            icode = icode_counter
            icode_counter += 1
            icode_map[key] = icode
            instruction_metas.append(InstructionMeta(icode=icode, itype=itype, src_tensors=src_tensors, dst_tensors=dst_tensors))
        else:
            icode = icode_map[key]

        src_bar_list = node_src_barriers.get(node.id, [])
        src_barriers = tuple(bid for bid, _ in src_bar_list)
        src_barrier_targets = tuple(tgt for _, tgt in src_bar_list)
        dst_barriers = tuple(node_dst_barriers.get(node.id, []))
        num_input_barriers = node_num_input_barriers.get(node.id, 0)
        num_reuse_barriers = node_num_reuse_barriers.get(node.id, 0)

        for block_index in itype.block_indices(src_metas, node.out_tensors):
            instructions.append(Instruction(
                icode=icode,
                src_tensors=src_tensors,
                dst_tensors=dst_tensors,
                indices=block_index,
                src_barriers=src_barriers,
                src_barrier_targets=src_barrier_targets,
                num_input_barriers=num_input_barriers,
                num_reuse_barriers=num_reuse_barriers,
                num_dst_barriers=len(dst_barriers),
                dst_barriers=dst_barriers,
            ))

        # Pad to CLUSTER_SIZE with noops so op boundaries are cluster-aligned
        while len(instructions) % Dispatcher.CLUSTER_SIZE != 0:
            instructions.append(noop)

    return (
        instruction_metas,
        tensor_metas,
        instructions,
        barrier_counter,
        tuple(input_tensor_indices),
        tuple(output_tensor_indices),
    )
