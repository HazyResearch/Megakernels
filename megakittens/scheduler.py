from __future__ import annotations

from typing import Dict, List, Tuple

from .dispatcher import Dispatcher
from .itypes.noop import Noop
from .jit.cuda_utils import get_sm_count
from .schema.dag import DAG
from .schema.tensor import TensorMeta
from .schema.instruction import (
    IType,
    Instruction,
    InstructionMeta,
)


# Derived from the instruction struct
MAX_BARRIERS = 2**32 - 1
MAX_TENSOR_ALLOCATIONS = 256


def _get_instruction_count_and_offset(
    dag: DAG,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Phase 1: Count instructions per node and compute cluster-aligned offsets.

    Args:
        dag: validated DAG of input, and output, and compute nodes.

    Returns:
        (node_inst_count, node_inst_offset) where:
        - node_inst_count: Dict[int, int], node_id -> number of instructions this node generates.
        - node_inst_offset: Dict[int, int], node_id -> starting index in the global instruction list.
    """
    node_inst_count: Dict[int, int] = {}
    node_inst_offset: Dict[int, int] = {}
    cumulative_offset = 0
    for node in dag.nodes:
        if node.is_input or node.is_output:
            continue
        if len(node.in_nodes) > Instruction.MAX_SRC_TENSORS:
            raise RuntimeError(
                f"[MegaKittens] Node has {len(node.in_nodes)} src tensors (max {Instruction.MAX_SRC_TENSORS})"
            )
        if len(node.out_tensors) > Instruction.MAX_DST_TENSORS:
            raise RuntimeError(
                f"[MegaKittens] Node has {len(node.out_tensors)} dst tensors (max {Instruction.MAX_DST_TENSORS})"
            )
        src_metas = tuple(in_node.out_tensors[slot_idx] for in_node, slot_idx in node.in_nodes)
        count = node.itype.num_instructions(src_metas, node.out_tensors)
        node_inst_count[node.id] = count
        node_inst_offset[node.id] = cumulative_offset
        cumulative_offset += count + (-count) % Dispatcher.CLUSTER_SIZE  # CTA pairs must get same instruction type
    return node_inst_count, node_inst_offset


def _assign_tensors(
    dag: DAG,
    node_inst_count: Dict[int, int],
    node_inst_offset: Dict[int, int],
) -> Tuple[List[TensorMeta], Dict[Tuple[int, int], int], List[int], List[int], List[Tuple[List[int], int, int]]]:
    """Phase 2: Assign tensor indices, reuse memory where possible, collect I/O indices.

    Args:
        dag: validated DAG of input, and output, and compute nodes.
        node_inst_count: Dict[int, int], node_id -> number of instructions this node generates.
        node_inst_offset: Dict[int, int], node_id -> starting index in the global instruction list.

    Returns:
        (tensor_metas, tensor_index, input_tensor_indices, output_tensor_indices, release_barriers) where:
        - tensor_metas: List[TensorMeta], flat list of unique tensor metadata.
        - tensor_index: Dict[Tuple[int, int], int], (node_id, output_slot) -> index into tensor_metas.
        - input_tensor_indices: List[int], tensor_metas indices for graph inputs, in order.
        - output_tensor_indices: List[int], tensor_metas indices for graph outputs, in order.
        - release_barriers: List[Tuple[List[int], int, int]], each is (consumer_node_ids, num_consumer_insts, reuser_node_id) for tensor reuse.
    """
    tensor_metas: List[TensorMeta] = []
    tensor_index: Dict[Tuple[int, int], int] = {}
    input_tensor_indices: List[int] = []
    output_tensor_indices: List[int] = []
    tensor_pool: Dict[TensorMeta, List[Tuple[List[int], int, int, int]]] = {}
    release_barriers: List[Tuple[List[int], int, int]] = []

    for node in dag.nodes:
        for out_idx, tensor_meta in enumerate(node.out_tensors):
            reused = False
            if not node.is_input and not node.is_output:
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
            if node.is_input or any(c.is_output for c in consumer_nodes):
                continue
            consumer_node_ids = [c.id for c in consumer_nodes]
            num_consumer_insts = sum(node_inst_count[cid] for cid in consumer_node_ids)
            last_consumer_inst = max(node_inst_offset[c.id] + node_inst_count[c.id] + (-node_inst_count[c.id]) % Dispatcher.CLUSTER_SIZE for c in consumer_nodes)
            tensor_pool.setdefault(tensor_meta, []).append((consumer_node_ids, last_consumer_inst, num_consumer_insts, tensor_index[(node.id, out_idx)]))

        if node.is_input:
            if len(node.out_tensors) != 1:
                raise RuntimeError(
                    f"[MegaKittens] Input node has {len(node.out_tensors)} outputs (expected 1)"
                )
            input_tensor_indices.append(tensor_index[(node.id, 0)])
        elif node.is_output:
            if len(output_tensor_indices) != 0:
                raise RuntimeError("[MegaKittens] Expected 1 output node")
            output_tensor_indices.extend(
                tensor_index[(in_node.id, slot_idx)] for in_node, slot_idx in node.in_nodes
            )
    if not input_tensor_indices:
        raise RuntimeError("[MegaKittens] Graph has no input tensors")
    if not output_tensor_indices:
        raise RuntimeError("[MegaKittens] Graph has no output tensors")

    return tensor_metas, tensor_index, input_tensor_indices, output_tensor_indices, release_barriers


def _assign_barriers(
    dag: DAG,
    node_inst_count: Dict[int, int],
    release_barriers: List[Tuple[List[int], int, int]],
) -> Tuple[Dict[int, List[int]], Dict[int, List[Tuple[int, int]]], Dict[int, int], Dict[int, int], int]:
    """Phase 3: Assign barriers for inter-node dependencies and tensor reuse."""
    barrier_counter = 0
    node_dst_barriers: Dict[int, List[int]] = {}
    node_src_barriers: Dict[int, List[Tuple[int, int]]] = {}
    node_num_input_barriers: Dict[int, int] = {}
    node_num_reuse_barriers: Dict[int, int] = {}

    # TODO: reuse barriers
    # Input dependency barriers
    for node in dag.nodes:
        if node.is_input or node.is_output:
            continue
        for in_node, _slot_idx in node.in_nodes:
            if in_node.is_input or in_node.is_output:
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

    return node_dst_barriers, node_src_barriers, node_num_input_barriers, node_num_reuse_barriers, barrier_counter


def _generate_instructions(
    dag: DAG,
    tensor_index: Dict[Tuple[int, int], int],
    node_block_indices: Dict[int, list],
    inst_dst_barriers: Dict[Tuple[int, int], List[int]],
    inst_src_barriers: Dict[Tuple[int, int], List[Tuple[int, int]]],
    inst_num_input_barriers: Dict[Tuple[int, int], int],
    inst_num_reuse_barriers: Dict[Tuple[int, int], int],
) -> Tuple[List[InstructionMeta], List[Instruction]]:
    """Phase 4: Build flat instruction list with icodes, barriers, and cluster-aligned padding.

    Args:
        dag: validated DAG of input, and output, and compute nodes.
        tensor_index: Dict[Tuple[int, int], int], (node_id, output_slot) -> index into tensor_metas.
        node_block_indices: Dict[int, list], node_id -> list of per-instruction tile coordinate tuples from block_indices().
        inst_dst_barriers: Dict[Tuple[int, int], List[int]], (node_id, local_inst_idx) -> barrier IDs this instruction arrives on after completion.
        inst_src_barriers: Dict[Tuple[int, int], List[Tuple[int, int]]], (node_id, local_inst_idx) -> (barrier_id, target_count) pairs to wait on before starting.
        inst_num_input_barriers: Dict[Tuple[int, int], int], (node_id, local_inst_idx) -> how many of src_barriers are data dependency barriers.
        inst_num_reuse_barriers: Dict[Tuple[int, int], int], (node_id, local_inst_idx) -> how many of src_barriers are tensor reuse barriers.

    Returns:
        (instruction_metas, instructions) where:
        - instruction_metas: List[InstructionMeta], one per unique icode. Index 0 is always noop.
        - instructions: List[Instruction], padded to CLUSTER_SIZE with noops.
    """
    instructions: List[Instruction] = []
    icode_counter = 1  # icode 0 is reserved for noop
    instruction_metas: List[InstructionMeta] = [InstructionMeta(icode=0, itype=Noop(), src_tensors=(), dst_tensors=())]
    icode_map: Dict[Tuple[IType, Tuple[int, ...], Tuple[int, ...]], int] = {}
    noop = Instruction(icode=0, src_tensors=(), dst_tensors=(), indices=(), src_barriers=(), src_barrier_targets=(), num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0, dst_barriers=())

    for node in dag.nodes:
        if node.is_input or node.is_output:
            continue

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
        node.itype.validate(src_metas, dst_metas)

        key = (node.itype, src_tensors, dst_tensors)
        if key not in icode_map:
            icode = icode_counter
            icode_counter += 1
            icode_map[key] = icode
            instruction_metas.append(InstructionMeta(icode=icode, itype=node.itype, src_tensors=src_tensors, dst_tensors=dst_tensors))
        else:
            icode = icode_map[key]

        for local_idx, block_index in enumerate(node_block_indices[node.id]):
            src_bar_list = inst_src_barriers.get((node.id, local_idx), [])
            dst_bariers = inst_dst_barriers.get((node.id, local_idx), [])
            num_input_barriers = inst_num_input_barriers.get((node.id, local_idx), 0)
            num_reuse_barriers = inst_num_reuse_barriers.get((node.id, local_idx), 0)

            instructions.append(Instruction(
                icode=icode,
                src_tensors=src_tensors,
                dst_tensors=dst_tensors,
                indices=block_index,
                src_barriers=tuple(bid for bid, _ in src_bar_list),
                src_barrier_targets=tuple(tgt for _, tgt in src_bar_list),
                num_input_barriers=num_input_barriers,
                num_reuse_barriers=num_reuse_barriers,
                num_dst_barriers=len(dst_bariers),
                dst_barriers=tuple(dst_bariers),
            ))

        # Pad to CLUSTER_SIZE with noops so op boundaries are cluster-aligned
        while len(instructions) % Dispatcher.CLUSTER_SIZE != 0:
            instructions.append(noop)

    return instruction_metas, instructions


def schedule(
    dag: DAG,
) -> Tuple[List[InstructionMeta], List[TensorMeta], List[Instruction], int, Tuple[int, ...], Tuple[int, ...]]:
    """Convert a validated DAG into a minimal set of tensors and a flat per-SM instruction list."""
    node_inst_count, node_inst_offset = _get_instruction_count_and_offset(dag)
    tensor_metas, tensor_index, input_tensor_indices, output_tensor_indices, release_barriers = _assign_tensors(dag, node_inst_count, node_inst_offset)
    node_dst_barriers, node_src_barriers, node_num_input_barriers, node_num_reuse_barriers, barrier_counter = _assign_barriers(dag, node_inst_count, release_barriers)
    instruction_metas, instructions = _generate_instructions(dag, tensor_index, node_dst_barriers, node_src_barriers, node_num_input_barriers, node_num_reuse_barriers)

    return (
        instruction_metas,
        tensor_metas,
        instructions,
        barrier_counter,
        tuple(input_tensor_indices),
        tuple(output_tensor_indices),
    )
