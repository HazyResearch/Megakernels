from __future__ import annotations

import itertools
from collections import defaultdict
from functools import reduce
from math import gcd
from typing import Dict, List, Tuple

from .dispatcher import Dispatcher
from .utils import timed
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


def get_instruction_count_and_offset(
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
        count = node.itype.num_instructions(src_metas, node.out_tensors, src_ranges=node.in_ranges, dst_ranges=node.out_ranges)
        node_inst_count[node.id] = count
        node_inst_offset[node.id] = cumulative_offset
        cumulative_offset += count + (-count) % Dispatcher.CLUSTER_SIZE  # CTA pairs must get same instruction type
    return node_inst_count, node_inst_offset


def assign_tensors(
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
        if node.is_output:
            if len(output_tensor_indices) != 0:
                raise RuntimeError("[MegaKittens] Expected 1 output node")
            output_tensor_indices.extend(tensor_index[(in_node.id, slot_idx)] for in_node, slot_idx in node.in_nodes)
            continue  # no need to allocate tensors for the output node

        inplace_mapping = node.itype.inplace_mapping if node.itype is not None else None
        for out_idx, tensor_meta in enumerate(node.out_tensors):
            reused = False
            if not node.is_input:
                if inplace_mapping is not None and out_idx in inplace_mapping:
                    in_node, in_slot = node.in_nodes[inplace_mapping[out_idx]]
                    tensor_index[(node.id, out_idx)] = tensor_index[(in_node.id, in_slot)]
                    reused = True
                else:
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
            if inplace_mapping is not None and out_idx in inplace_mapping:
                continue  # in-place outputs share storage with an input; don't add a duplicate pool entry
            consumer_node_ids = [c.id for c in consumer_nodes]
            num_consumer_insts = sum(node_inst_count[cid] for cid in consumer_node_ids)
            last_consumer_inst = max(node_inst_offset[c.id] + node_inst_count[c.id] + (-node_inst_count[c.id]) % Dispatcher.CLUSTER_SIZE for c in consumer_nodes)
            tensor_pool.setdefault(tensor_meta, []).append((consumer_node_ids, last_consumer_inst, num_consumer_insts, tensor_index[(node.id, out_idx)]))

        if node.is_input:
            if len(node.out_tensors) != 1:
                raise RuntimeError(f"[MegaKittens] Input node has {len(node.out_tensors)} outputs (expected 1)")
            input_tensor_indices.append(tensor_index[(node.id, 0)])

    if not input_tensor_indices:
        raise RuntimeError("[MegaKittens] Graph has no input tensors")
    if not output_tensor_indices:
        raise RuntimeError("[MegaKittens] Graph has no output tensors")

    return tensor_metas, tensor_index, input_tensor_indices, output_tensor_indices, release_barriers


def assign_barriers(
    dag: DAG,
    node_inst_count: Dict[int, int],
    release_barriers: List[Tuple[List[int], int, int]],
) -> Tuple[Dict[int, list], Dict[int, list], Dict[int, list], Dict[int, list], Dict[Tuple[int, int], List[int]], Dict[Tuple[int, int], List[Tuple[int, int]]], Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], int]:
    """Phase 3: Compute fine-grained per-instruction barriers via tile region overlap.

    Args:
        dag: validated DAG of input, and output, and compute nodes.
        node_inst_count: Dict[int, int], node_id -> number of instructions this node generates.
        release_barriers: List[Tuple[List[int], int, int]], each is (consumer_node_ids, num_consumer_insts, reuser_node_id) for tensor reuse.

    Returns:
        (node_block_indices, inst_dst_barriers, inst_src_barriers, inst_num_input_barriers, inst_num_reuse_barriers, barrier_counter) where:
        - node_block_indices: Dict[int, list], node_id -> list of per-instruction tile coordinate tuples from block_indices().
        - inst_dst_barriers: Dict[Tuple[int, int], List[int]], (node_id, local_inst_idx) -> barrier IDs this instruction arrives on after completion.
        - inst_src_barriers: Dict[Tuple[int, int], List[Tuple[int, int]]], (node_id, local_inst_idx) -> (barrier_id, target_count) pairs to wait on before starting.
        - inst_num_input_barriers: Dict[Tuple[int, int], int], (node_id, local_inst_idx) -> how many of src_barriers are data dependency barriers.
        - inst_num_reuse_barriers: Dict[Tuple[int, int], int], (node_id, local_inst_idx) -> how many of src_barriers are tensor reuse barriers.
        - barrier_counter: int, total number of barriers allocated.
    """
    barrier_counter = 0

    # Per-instruction barriers, keyed by (node_id, local_inst_idx)
    inst_src_barriers: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    inst_dst_barriers: Dict[Tuple[int, int], List[int]] = {}
    inst_num_input_barriers: Dict[Tuple[int, int], int] = {}
    inst_num_reuse_barriers: Dict[Tuple[int, int], int] = {}

    # Step 1. Collect per-instruction tile regions for each compute node
    node_regions: Dict[int, list] = {}  # node_id -> list of (src_regions, dst_regions) per instruction
    node_block_indices: Dict[int, list] = {}
    for node in dag.nodes:
        if node.is_input or node.is_output:
            continue
        src_metas = tuple(in_node.out_tensors[slot_idx] for in_node, slot_idx in node.in_nodes)
        block_indices = node.itype.block_indices(src_metas, node.out_tensors, src_ranges=node.in_ranges, dst_ranges=node.out_ranges)
        node_block_indices[node.id] = block_indices
        node_regions[node.id] = [
            node.itype.access_regions(block_index, src_metas, node.out_tensors)
            for block_index in block_indices
        ]

    # Step 2. Input dependency barriers
    dependency_map: Dict[Tuple[int, frozenset[int]], Dict[int, List[int]]] = {}  # (producer node ID, set of local indices) -> { (consumer node ID) -> (list of local indices) }
    for node in dag.nodes:
        if node.is_input or node.is_output:
            continue
        for edge_idx, (in_node, slot_idx) in enumerate(node.in_nodes):  # Combined with the outer loop, O(num_edges)
            if in_node.is_input or in_node.is_output:
                continue

            # Collect all regions produced/consumed by this edge
            producer_regions = [dst_regions[slot_idx] for _, dst_regions in node_regions[in_node.id]]
            consumer_regions = [src_regions[edge_idx] for src_regions, _ in node_regions[node.id]]

            ndim = len(producer_regions[0])
            unit_region: List[int] = []
            for d in range(ndim):
                all_sizes: set[int] = set()
                for region in producer_regions:
                    all_sizes.add(region[d][1] - region[d][0])
                for region in consumer_regions:
                    all_sizes.add(region[d][1] - region[d][0])
                unit_region.append(reduce(gcd, all_sizes))

            unit_region_index_to_p_local_index: Dict[tuple, List[int]] = defaultdict(list)
            for p_local_index, region in enumerate(producer_regions):
                unit_region_indices = itertools.product(*[range(region[d][0] // unit_region[d], region[d][1] // unit_region[d]) for d in range(ndim)])
                for unit_region_index in unit_region_indices:
                    unit_region_index_to_p_local_index[unit_region_index].append(p_local_index)

            c_region_cache: Dict[tuple, frozenset[int]] = {}
            for c_local_index, region in enumerate(consumer_regions):
                if region not in c_region_cache:
                    unit_region_indices = itertools.product(*[range(region[d][0] // unit_region[d], region[d][1] // unit_region[d]) for d in range(ndim)])
                    matching_p_local_indices: set[int] = set()
                    for unit_region_index in unit_region_indices:
                        if unit_region_index not in unit_region_index_to_p_local_index:
                            raise RuntimeError("[MegaKittens] Matching producer region not found.")
                        matching_p_local_indices.update(unit_region_index_to_p_local_index[unit_region_index])
                    c_region_cache[region] = frozenset(matching_p_local_indices)
                dependency_map.setdefault((in_node.id, c_region_cache[region]), {}).setdefault(node.id, []).append(c_local_index)

    # Each unique (producer_node_id, dependency set) becomes one barrier
    for (producer_id, dependent_p_local_indices_set), consumers_by_node in dependency_map.items():
        if not dependent_p_local_indices_set:
            continue
        if barrier_counter >= MAX_BARRIERS:
            raise RuntimeError(f"[MegaKittens] Barrier count exceeds {MAX_BARRIERS}")
        bid = barrier_counter
        barrier_counter += 1
        target = len(dependent_p_local_indices_set)
        for p_local_index in dependent_p_local_indices_set:
            inst_dst_barriers.setdefault((producer_id, p_local_index), []).append(bid)
        for consumer_id, c_local_indices in consumers_by_node.items():
            for c_local_index in c_local_indices:
                inst_src_barriers.setdefault((consumer_id, c_local_index), []).append((bid, target))
                inst_num_input_barriers[(consumer_id, c_local_index)] = inst_num_input_barriers.get((consumer_id, c_local_index), 0) + 1

    # Step 3. Tensor reuse barriers
    for consumer_node_ids, num_consumer_insts, reuser_id in release_barriers:
        if barrier_counter >= MAX_BARRIERS:
            raise RuntimeError(f"[MegaKittens] Barrier count exceeds {MAX_BARRIERS}")
        bid = barrier_counter
        barrier_counter += 1
        for nid in consumer_node_ids:
            for local_index in range(node_inst_count[nid]):
                inst_dst_barriers.setdefault((nid, local_index), []).append(bid)
        for local_index in range(node_inst_count[reuser_id]):
            inst_src_barriers.setdefault((reuser_id, local_index), []).append((bid, num_consumer_insts))
            inst_num_reuse_barriers[(reuser_id, local_index)] = inst_num_reuse_barriers.get((reuser_id, local_index), 0) + 1

    # Step 4. Validate per-instruction barrier limits
    for key, barriers in inst_dst_barriers.items():
        if len(barriers) > Instruction.MAX_DST_BARRIERS:
            raise RuntimeError(
                f"[MegaKittens] Instruction {key} has {len(barriers)} dst barriers (max {Instruction.MAX_DST_BARRIERS})"
            )
    for key, barriers in inst_src_barriers.items():
        if len(barriers) > Instruction.MAX_SRC_BARRIERS:
            raise RuntimeError(
                f"[MegaKittens] Instruction {key} has {len(barriers)} src barriers (max {Instruction.MAX_SRC_BARRIERS})"
            )

    return node_block_indices, inst_dst_barriers, inst_src_barriers, inst_num_input_barriers, inst_num_reuse_barriers, barrier_counter


def generate_instructions(
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
        node.itype.validate(src_metas, dst_metas, src_ranges=node.in_ranges, dst_ranges=node.out_ranges)

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
    verbose: bool = True,
) -> Tuple[List[InstructionMeta], List[TensorMeta], List[Instruction], int, Tuple[int, ...], Tuple[int, ...]]:
    """Convert a validated DAG into a minimal set of tensors, barriers, and a flat per-SM instruction list.

    Args:
        dag: validated DAG of input, output, and compute nodes.

    Returns:
        (instruction_metas, tensor_metas, instructions, barrier_counter, input_tensor_indices, output_tensor_indices) where:
        - instruction_metas: List[InstructionMeta], one per unique instruction. Index 0 is always noop.
        - tensor_metas: List[TensorMeta], flat list of unique tensor metadata.
        - instructions: List[Instruction], padded to CLUSTER_SIZE with noops.
        - barrier_counter: int, total number of barriers allocated.
        - input_tensor_indices: Tuple[int, ...], tensor_metas indices for graph inputs, in order.
        - output_tensor_indices: Tuple[int, ...], tensor_metas indices for graph outputs, in order.
    """
    with timed("[Scheduler] Counted instructions and offsets", verbose):
        node_inst_count, node_inst_offset = get_instruction_count_and_offset(dag)
    with timed("[Scheduler] Assigned tensors", verbose):
        tensor_metas, tensor_index, input_tensor_indices, output_tensor_indices, release_barriers = assign_tensors(dag, node_inst_count, node_inst_offset)
    with timed("[Scheduler] Assigned barriers", verbose):
        node_block_indices, inst_dst_barriers, inst_src_barriers, inst_num_input_barriers, inst_num_reuse_barriers, barrier_counter = assign_barriers(dag, node_inst_count, release_barriers)
    with timed("[Scheduler] Generated instructions", verbose):
        instruction_metas, instructions = generate_instructions(dag, tensor_index, node_block_indices, inst_dst_barriers, inst_src_barriers, inst_num_input_barriers, inst_num_reuse_barriers)

    return (
        instruction_metas,
        tensor_metas,
        instructions,
        barrier_counter,
        tuple(input_tensor_indices),
        tuple(output_tensor_indices),
    )
