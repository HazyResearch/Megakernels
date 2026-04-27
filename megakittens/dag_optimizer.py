from __future__ import annotations

from .schema.dag import DAG, Node
from .schema.tensor import TensorRange


def prune_dead_nodes(dag: DAG) -> None:
    """Remove nodes not reachable from the output."""
    output_nodes = [n for n in dag.nodes if n.is_output]
    if len(output_nodes) != 1:
        raise RuntimeError(f"[MegaKittens] Expected 1 output node, got {len(output_nodes)}")

    alive: set[int] = set()
    stack = [output_nodes[0]]
    while stack:
        node = stack.pop()
        if node.id in alive:
            continue
        alive.add(node.id)
        for in_node, _ in node.in_nodes:
            stack.append(in_node)

    if len(alive) == len(dag.nodes):
        return

    dag.nodes = [n for n in dag.nodes if n.id in alive]
    for node in dag.nodes:
        for slot_idx in range(len(node.out_nodes)):
            node.out_nodes[slot_idx][:] = [n for n in node.out_nodes[slot_idx] if n.id in alive]


def merge_getitem_select_slice_ops(dag: DAG) -> None:
    """Collapses the getitem, select, and slice nodes from the DAG."""
    view_node_ids: set[int] = set()

    for node in dag.nodes:
        if node.itype not in ("getitem", "select", "slice"):
            continue
        view_node_ids.add(node.id)

        if len(node.in_nodes) != 1:
            raise RuntimeError(
                f"[MegaKittens] View node (itype={node.itype!r}) must have exactly 1 input, got {len(node.in_nodes)}"
            )
        source, source_slot = node.in_nodes[0]
        view_in_range = node.in_ranges[0]

        for consumer in list(node.out_nodes[0]):
            new_in_nodes = list(consumer.in_nodes)
            new_in_ranges = list(consumer.in_ranges)
            for edge_idx in range(len(new_in_nodes)):
                in_node, in_slot = new_in_nodes[edge_idx]
                if in_node.id != node.id or in_slot != 0:
                    continue
                new_in_nodes[edge_idx] = (source, source_slot)
                new_in_ranges[edge_idx] = TensorRange.compose(view_in_range, consumer.in_ranges[edge_idx])
                source.out_nodes[source_slot].append(consumer)
            consumer.in_nodes = tuple(new_in_nodes)
            consumer.in_ranges = tuple(new_in_ranges)

    if not view_node_ids:
        return

    dag.nodes = [n for n in dag.nodes if n.id not in view_node_ids]
    for node in dag.nodes:
        for slot_idx in range(len(node.out_nodes)):
            node.out_nodes[slot_idx][:] = [n for n in node.out_nodes[slot_idx] if n.id not in view_node_ids]

    for node in dag.nodes:
        if node.is_output:
            for edge_idx, (in_node, in_slot) in enumerate(node.in_nodes):
                src_shape = in_node.out_tensors[in_slot].shape
                if not node.in_ranges[edge_idx].is_full(src_shape):
                    raise RuntimeError(
                        "[MegaKittens] Output node directly consumes a sliced view;"
                        " returning sliced views is not supported"
                    )


def optimize_dag(dag: DAG) -> DAG:
    """Apply optimization passes to the DAG before scheduling."""
    prune_dead_nodes(dag)
    merge_getitem_select_slice_ops(dag)

    return dag
