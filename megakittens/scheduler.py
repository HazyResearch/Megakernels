from __future__ import annotations

import itertools
from collections import defaultdict
from functools import reduce
from math import gcd
from typing import Dict, List, Tuple

from .utils import timed
from .itypes.noop import Noop
from .jit.cuda_utils import get_sm_count
from .schema.dag import DAG
from .schema.device import Device
from .schema.tensor import TensorMeta, TensorStorage
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
    cluster_size: int,
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
        count = node.itype.num_instructions(node.in_tensors, node.out_tensors, src_ranges=node.in_ranges, dst_ranges=node.out_ranges)
        node_inst_count[node.id] = count
        node_inst_offset[node.id] = cumulative_offset
        cumulative_offset += count + (-count) % cluster_size  # CTA pairs must get same instruction type
    return node_inst_count, node_inst_offset


def assign_tensors(
    dag: DAG,
    cluster_size: int,
    node_inst_count: Dict[int, int],
    node_inst_offset: Dict[int, int],
) -> Tuple[List[TensorMeta], Dict[Tuple[int, str, int], int], List[int], List[int], List[Tuple[List[int], int, int]]]:
    """Phase 2: Assign tensor indices, reuse memory where possible, collect I/O indices.

    Args:
        dag: validated DAG of input, and output, and compute nodes.
        node_inst_count: Dict[int, int], node_id -> number of instructions this node generates.
        node_inst_offset: Dict[int, int], node_id -> starting index in the global instruction list.

    Returns:
        (tensor_metas, tensor_index, input_tensor_indices, output_tensor_indices, release_barriers) where:
        - tensor_metas: List[TensorMeta], flat list of unique tensor metadata.
        - tensor_index: Dict[Tuple[int, str, int], int], (node_id, "in"/"out", slot_idx) -> index into tensor_metas.
        - input_tensor_indices: List[int], tensor_metas indices for graph inputs, in order.
        - output_tensor_indices: List[int], tensor_metas indices for graph outputs, in order.
        - release_barriers: List[Tuple[List[int], int, int]], each is (consumer_node_ids, num_consumer_insts, reuser_node_id) for tensor reuse.
    """
    tensor_metas: List[TensorMeta] = []
    tensor_index: Dict[Tuple[int, str, int], int] = {}
    input_tensor_indices: List[int] = []
    output_tensor_indices: List[int] = []
    storage_pool: Dict[Tuple[int, Device], List[Tuple[List[int], int, int, int]]] = {}
    storage_users: Dict[int, List[int]] = {}  # storage.id -> list of tensor_meta indices
    release_barriers: List[Tuple[List[int], int, int]] = []

    for node in dag.nodes:
        if node.is_output:
            if len(output_tensor_indices) != 0:
                raise RuntimeError("[MegaKittens] Expected 1 output node")
            output_tensor_indices.extend(tensor_index[(in_node.id, "out", slot_idx)] for in_node, slot_idx in node.in_nodes)
            continue  # no need to allocate tensors for the output node

        inplace_mapping = node.itype.inplace_mapping if node.itype is not None else None
        for out_idx, tensor_meta in enumerate(node.out_tensors):
            reused = False
            if not node.is_input:
                if inplace_mapping is not None and out_idx in inplace_mapping:
                    in_node, in_slot = node.in_nodes[inplace_mapping[out_idx]]
                    tensor_index[(node.id, "out", out_idx)] = tensor_index[(in_node.id, "out", in_slot)]
                    reused = True
                    other_consumer_ids = ([in_node.id] if not in_node.is_input else []) + [consumer.id for consumer in in_node.out_nodes[in_slot] if consumer.id != node.id and not consumer.is_output]
                    if other_consumer_ids:
                        num_other_consumer_insts = sum(node_inst_count[cid] for cid in other_consumer_ids)
                        release_barriers.append((other_consumer_ids, num_other_consumer_insts, node.id))
                else:
                    pool_key = (tensor_meta.size_bytes, tensor_meta.device)
                    free_tensors = storage_pool.get(pool_key, [])
                    for i, (consumer_node_ids, last_consumer_inst, num_consumer_insts, storage_id) in enumerate(free_tensors):
                        if node_inst_offset[node.id] - last_consumer_inst >= get_sm_count():
                            matched_tid = None
                            for t in storage_users[storage_id]:
                                if (tensor_metas[t].dtype == tensor_meta.dtype and tensor_metas[t].shape == tensor_meta.shape and tensor_metas[t].device == tensor_meta.device):
                                    matched_tid = t
                                    break
                            if matched_tid is not None:
                                tensor_index[(node.id, "out", out_idx)] = matched_tid
                            else:
                                if len(tensor_metas) >= MAX_TENSOR_ALLOCATIONS:
                                    raise RuntimeError(
                                        f"[MegaKittens] The given compute graph requires tensor count exceeding {MAX_TENSOR_ALLOCATIONS}."
                                    )
                                tensor_index[(node.id, "out", out_idx)] = len(tensor_metas)
                                storage_users[storage_id].append(len(tensor_metas))
                                tensor_metas.append(TensorMeta(dtype=tensor_meta.dtype, shape=tensor_meta.shape, device=tensor_meta.device, storage=tensor_metas[storage_users[storage_id][0]].storage))
                            free_tensors.pop(i)
                            release_barriers.append((consumer_node_ids, num_consumer_insts, node.id))
                            reused = True
                            break

            if not reused:
                if len(tensor_metas) >= MAX_TENSOR_ALLOCATIONS:
                    raise RuntimeError(
                        f"[MegaKittens] The given compute graph requires tensor count exceeding {MAX_TENSOR_ALLOCATIONS}."
                    )
                storage = TensorStorage(size=tensor_meta.size_bytes, device=tensor_meta.device)
                tensor_meta = TensorMeta(dtype=tensor_meta.dtype, shape=tensor_meta.shape, device=tensor_meta.device, storage=storage)
                tensor_index[(node.id, "out", out_idx)] = len(tensor_metas)
                storage_users[storage.id] = [len(tensor_metas)]
                tensor_metas.append(tensor_meta)

            consumer_nodes = [node] + node.out_nodes[out_idx]
            if node.is_input or any(c.is_output for c in consumer_nodes):
                continue
            if any(  # if any consumer is in-place-mutating this, then don't put this to reuse pool
                c.itype is not None and c.itype.inplace_mapping is not None and
                any(c.in_nodes[c_in_idx][0].id == node.id and c.in_nodes[c_in_idx][1] == out_idx for c_in_idx in c.itype.inplace_mapping.values())
                for c in node.out_nodes[out_idx]
            ):
                continue
            consumer_node_ids = [c.id for c in consumer_nodes]
            num_consumer_insts = sum(node_inst_count[cid] for cid in consumer_node_ids)
            last_consumer_inst = max(node_inst_offset[c.id] + node_inst_count[c.id] + (-node_inst_count[c.id]) % cluster_size for c in consumer_nodes)
            pool_key = (tensor_meta.size_bytes, tensor_meta.device)
            storage_id = tensor_metas[tensor_index[(node.id, "out", out_idx)]].storage.id
            storage_pool.setdefault(pool_key, []).append((consumer_node_ids, last_consumer_inst, num_consumer_insts, storage_id))

        if not node.is_input:
            for edge_idx, ((source_node, source_out_slot), in_tensor) in enumerate(zip(node.in_nodes, node.in_tensors)):
                source_tensor_meta = tensor_index[(source_node.id, "out", source_out_slot)]
                if in_tensor.shape == tensor_metas[source_tensor_meta].shape:
                    tensor_index[(node.id, "in", edge_idx)] = source_tensor_meta
                else:
                    if len(tensor_metas) >= MAX_TENSOR_ALLOCATIONS:
                        raise RuntimeError(f"[MegaKittens] Tensor count exceeds {MAX_TENSOR_ALLOCATIONS}.")
                    storage = tensor_metas[source_tensor_meta].storage
                    tensor_index[(node.id, "in", edge_idx)] = len(tensor_metas)
                    storage_users[storage.id].append(len(tensor_metas))
                    tensor_metas.append(TensorMeta(dtype=in_tensor.dtype, shape=in_tensor.shape, device=in_tensor.device, storage=storage))
        else:
            if len(node.out_tensors) != 1:
                raise RuntimeError(f"[MegaKittens] Input node has {len(node.out_tensors)} outputs (expected 1)")
            input_tensor_indices.append(tensor_index[(node.id, "out", 0)])

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
        (node_block_indices, inst_dst_barriers, inst_src_barriers, inst_num_src_input_barriers, inst_num_src_reuse_barriers, inst_num_dst_input_barriers, inst_num_dst_reuse_barriers, barrier_counter) where:
        - node_block_indices: Dict[int, list], node_id -> list of per-instruction tile coordinate tuples from block_indices().
        - inst_dst_barriers: Dict[Tuple[int, int], List[int]], (node_id, local_inst_idx) -> barrier IDs this instruction arrives on after completion.
        - inst_src_barriers: Dict[Tuple[int, int], List[Tuple[int, int]]], (node_id, local_inst_idx) -> (barrier_id, target_count) pairs to wait on before starting.
        - inst_num_src_input_barriers: Dict[Tuple[int, int], int], (node_id, local_inst_idx) -> how many of src_barriers are data dependency barriers.
        - inst_num_src_reuse_barriers: Dict[Tuple[int, int], int], (node_id, local_inst_idx) -> how many of src_barriers are tensor reuse barriers.
        - inst_num_dst_input_barriers: Dict[Tuple[int, int], int], (node_id, local_inst_idx) -> how many of dst_barriers are data dependency barriers.
        - inst_num_dst_reuse_barriers: Dict[Tuple[int, int], int], (node_id, local_inst_idx) -> how many of dst_barriers are tensor reuse barriers.
        - barrier_counter: int, total number of barriers allocated.
    """
    barrier_counter = 0

    # Per-instruction barriers, keyed by (node_id, local_inst_idx)
    inst_src_barriers: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    inst_dst_barriers: Dict[Tuple[int, int], List[int]] = {}
    inst_num_src_input_barriers: Dict[Tuple[int, int], int] = {}
    inst_num_src_reuse_barriers: Dict[Tuple[int, int], int] = {}
    inst_num_dst_input_barriers: Dict[Tuple[int, int], int] = {}
    inst_num_dst_reuse_barriers: Dict[Tuple[int, int], int] = {}

    # Step 1. Collect per-instruction tile regions for each compute node
    node_regions: Dict[int, list] = {}  # node_id -> list of (src_regions, dst_regions) per instruction
    node_block_indices: Dict[int, list] = {}
    for node in dag.nodes:
        if node.is_input or node.is_output:
            continue
        block_indices = node.itype.block_indices(node.in_tensors, node.out_tensors, src_ranges=node.in_ranges, dst_ranges=node.out_ranges)
        node_block_indices[node.id] = block_indices
        node_regions[node.id] = [
            node.itype.access_regions(block_index, node.in_tensors, node.out_tensors)
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

            ndim = len(producer_regions[0][0])
            unit_region: List[int] = []
            for d in range(ndim):
                all_sizes: set[int] = set()
                for boxes in producer_regions:
                    for box in boxes:
                        all_sizes.add(box[d][1] - box[d][0])
                for boxes in consumer_regions:
                    for box in boxes:
                        all_sizes.add(box[d][1] - box[d][0])
                unit_region.append(reduce(gcd, all_sizes))

            unit_region_index_to_p_local_index: Dict[tuple, List[int]] = defaultdict(list)
            for p_local_index, boxes in enumerate(producer_regions):
                for box in boxes:
                    for unit_region_index in itertools.product(*[range(box[d][0] // unit_region[d], box[d][1] // unit_region[d]) for d in range(ndim)]):
                        unit_region_index_to_p_local_index[unit_region_index].append(p_local_index)

            c_region_cache: Dict[tuple, frozenset[int]] = {}
            for c_local_index, boxes in enumerate(consumer_regions):
                cache_key = tuple(sorted(boxes))
                if cache_key not in c_region_cache:
                    matching_p_local_indices: set[int] = set()
                    for box in boxes:
                        for unit_region_index in itertools.product(*[range(box[d][0] // unit_region[d], box[d][1] // unit_region[d]) for d in range(ndim)]):
                            if unit_region_index not in unit_region_index_to_p_local_index:
                                raise RuntimeError("[MegaKittens] Matching producer region not found.")
                            matching_p_local_indices.update(unit_region_index_to_p_local_index[unit_region_index])
                    c_region_cache[cache_key] = frozenset(matching_p_local_indices)
                dependency_map.setdefault((in_node.id, c_region_cache[cache_key]), {}).setdefault(node.id, []).append(c_local_index)

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
            inst_num_dst_input_barriers[(producer_id, p_local_index)] = inst_num_dst_input_barriers.get((producer_id, p_local_index), 0) + 1
        for consumer_id, c_local_indices in consumers_by_node.items():
            for c_local_index in c_local_indices:
                inst_src_barriers.setdefault((consumer_id, c_local_index), []).append((bid, target))
                inst_num_src_input_barriers[(consumer_id, c_local_index)] = inst_num_src_input_barriers.get((consumer_id, c_local_index), 0) + 1

    # Step 3. Tensor reuse barriers
    for consumer_node_ids, num_consumer_insts, reuser_id in release_barriers:
        if barrier_counter >= MAX_BARRIERS:
            raise RuntimeError(f"[MegaKittens] Barrier count exceeds {MAX_BARRIERS}")
        bid = barrier_counter
        barrier_counter += 1
        for nid in consumer_node_ids:
            for local_index in range(node_inst_count[nid]):
                inst_dst_barriers.setdefault((nid, local_index), []).append(bid)
                inst_num_dst_reuse_barriers[(nid, local_index)] = inst_num_dst_reuse_barriers.get((nid, local_index), 0) + 1
        for local_index in range(node_inst_count[reuser_id]):
            inst_src_barriers.setdefault((reuser_id, local_index), []).append((bid, num_consumer_insts))
            inst_num_src_reuse_barriers[(reuser_id, local_index)] = inst_num_src_reuse_barriers.get((reuser_id, local_index), 0) + 1

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

    return node_block_indices, inst_dst_barriers, inst_src_barriers, inst_num_src_input_barriers, inst_num_src_reuse_barriers, inst_num_dst_input_barriers, inst_num_dst_reuse_barriers, barrier_counter


def generate_instructions(
    dag: DAG,
    cluster_size: int,
    tensor_index: Dict[Tuple[int, str, int], int],
    node_block_indices: Dict[int, list],
    inst_dst_barriers: Dict[Tuple[int, int], List[int]],
    inst_src_barriers: Dict[Tuple[int, int], List[Tuple[int, int]]],
    inst_num_src_input_barriers: Dict[Tuple[int, int], int],
    inst_num_src_reuse_barriers: Dict[Tuple[int, int], int],
    inst_num_dst_input_barriers: Dict[Tuple[int, int], int],
    inst_num_dst_reuse_barriers: Dict[Tuple[int, int], int],
) -> Tuple[List[InstructionMeta], List[Instruction]]:
    """Phase 4: Build flat instruction list with icodes, barriers, and cluster-aligned padding.

    Args:
        dag: validated DAG of input, and output, and compute nodes.
        tensor_index: Dict[Tuple[int, str, int], int], (node_id, "in"/"out", slot_idx) -> index into tensor_metas.
        node_block_indices: Dict[int, list], node_id -> list of per-instruction tile coordinate tuples from block_indices().
        inst_dst_barriers: Dict[Tuple[int, int], List[int]], (node_id, local_inst_idx) -> barrier IDs this instruction arrives on after completion.
        inst_src_barriers: Dict[Tuple[int, int], List[Tuple[int, int]]], (node_id, local_inst_idx) -> (barrier_id, target_count) pairs to wait on before starting.
        inst_num_src_input_barriers: Dict[Tuple[int, int], int], (node_id, local_inst_idx) -> how many of src_barriers are data dependency barriers.
        inst_num_src_reuse_barriers: Dict[Tuple[int, int], int], (node_id, local_inst_idx) -> how many of src_barriers are tensor reuse barriers.
        inst_num_dst_input_barriers: Dict[Tuple[int, int], int], (node_id, local_inst_idx) -> how many of dst_barriers are data dependency barriers.
        inst_num_dst_reuse_barriers: Dict[Tuple[int, int], int], (node_id, local_inst_idx) -> how many of dst_barriers are tensor reuse barriers.

    Returns:
        (instruction_metas, instructions) where:
        - instruction_metas: List[InstructionMeta], one per unique icode. Index 0 is always noop.
        - instructions: List[Instruction], padded to CLUSTER_SIZE with noops.
    """
    instructions: List[Instruction] = []
    icode_counter = 1  # icode 0 is reserved for noop
    instruction_metas: List[InstructionMeta] = [InstructionMeta(icode=0, itype=Noop(), src_tensors=(), dst_tensors=())]
    icode_map: Dict[Tuple[IType, Tuple[int, ...], Tuple[int, ...]], int] = {}
    noop = Instruction(icode=0, src_tensors=(), dst_tensors=(), indices=(), src_barriers=(), src_barrier_targets=(), num_src_input_barriers=0, num_src_reuse_barriers=0, num_dst_input_barriers=0, num_dst_reuse_barriers=0, dst_barriers=())

    for node in dag.nodes:
        if node.is_input or node.is_output:
            continue

        src_tensors = tuple(tensor_index[(node.id, "in", edge_idx)] for edge_idx in range(len(node.in_nodes)))

        dst_tensors = tuple(
            tensor_index[(node.id, "out", slot)]
            for slot in range(len(node.out_tensors))
        )

        node.itype.validate(node.in_tensors, node.out_tensors, src_ranges=node.in_ranges, dst_ranges=node.out_ranges)

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
            num_src_input_barriers = inst_num_src_input_barriers.get((node.id, local_idx), 0)
            num_src_reuse_barriers = inst_num_src_reuse_barriers.get((node.id, local_idx), 0)
            num_dst_input_barriers = inst_num_dst_input_barriers.get((node.id, local_idx), 0)
            num_dst_reuse_barriers = inst_num_dst_reuse_barriers.get((node.id, local_idx), 0)

            instructions.append(Instruction(
                icode=icode,
                src_tensors=src_tensors,
                dst_tensors=dst_tensors,
                indices=block_index,
                src_barriers=tuple(bid for bid, _ in src_bar_list),
                src_barrier_targets=tuple(tgt for _, tgt in src_bar_list),
                num_src_input_barriers=num_src_input_barriers,
                num_src_reuse_barriers=num_src_reuse_barriers,
                num_dst_input_barriers=num_dst_input_barriers,
                num_dst_reuse_barriers=num_dst_reuse_barriers,
                dst_barriers=tuple(dst_bariers),
            ))

        # Pad to cluster_size with noops so op boundaries are cluster-aligned
        while len(instructions) % cluster_size != 0:
            instructions.append(noop)

    return instruction_metas, instructions


def schedule(
    dag: DAG,
    cluster_size: int = 2,
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
        node_inst_count, node_inst_offset = get_instruction_count_and_offset(dag, cluster_size)
    with timed("[Scheduler] Assigned tensors", verbose):
        tensor_metas, tensor_index, input_tensor_indices, output_tensor_indices, release_barriers = assign_tensors(dag, cluster_size, node_inst_count, node_inst_offset)
    with timed("[Scheduler] Assigned barriers", verbose):
        node_block_indices, inst_dst_barriers, inst_src_barriers, inst_num_src_input_barriers, inst_num_src_reuse_barriers, inst_num_dst_input_barriers, inst_num_dst_reuse_barriers, barrier_counter = assign_barriers(dag, node_inst_count, release_barriers)
    with timed("[Scheduler] Generated instructions", verbose):
        instruction_metas, instructions = generate_instructions(dag, cluster_size, tensor_index, node_block_indices, inst_dst_barriers, inst_src_barriers, inst_num_src_input_barriers, inst_num_src_reuse_barriers, inst_num_dst_input_barriers, inst_num_dst_reuse_barriers)

    return (
        instruction_metas,
        tensor_metas,
        instructions,
        barrier_counter,
        tuple(input_tensor_indices),
        tuple(output_tensor_indices),
    )
