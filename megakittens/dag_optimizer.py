from __future__ import annotations

from .itypes.gemm import Gemm
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
            new_in_tensors = list(consumer.in_tensors)
            for edge_idx in range(len(new_in_nodes)):
                in_node, in_slot = new_in_nodes[edge_idx]
                if in_node.id != node.id or in_slot != 0:
                    continue
                new_in_nodes[edge_idx] = (source, source_slot)
                new_in_ranges[edge_idx] = TensorRange.compose(view_in_range, consumer.in_ranges[edge_idx])
                new_in_tensors[edge_idx] = source.out_tensors[source_slot]
                source.out_nodes[source_slot].append(consumer)
            consumer.in_nodes = tuple(new_in_nodes)
            consumer.in_ranges = tuple(new_in_ranges)
            consumer.in_tensors = tuple(new_in_tensors)

    if not view_node_ids:
        return

    dag.nodes = [n for n in dag.nodes if n.id not in view_node_ids]
    for node in dag.nodes:
        for slot_idx in range(len(node.out_nodes)):
            node.out_nodes[slot_idx][:] = [n for n in node.out_nodes[slot_idx] if n.id not in view_node_ids]

    for node in dag.nodes:
        if node.is_output:
            for edge_idx, in_tensor in enumerate(node.in_tensors):
                if not node.in_ranges[edge_idx].is_full(in_tensor.shape):
                    raise RuntimeError(
                        "[MegaKittens] Output node directly consumes a sliced view;"
                        " returning sliced views is not supported"
                    )


def merge_view_ops(dag: DAG) -> None:
    """Collapses view (aten.view/reshape/_unsafe_view) nodes from the DAG."""
    view_node_ids: set[int] = set()

    for node in dag.nodes:
        if node.itype != "view":
            continue
        view_node_ids.add(node.id)

        if len(node.in_nodes) != 1:
            raise RuntimeError(
                f"[MegaKittens] View node must have exactly 1 input, got {len(node.in_nodes)}"
            )
        source, source_slot = node.in_nodes[0]
        view_shape = node.out_tensors[0]

        for consumer in list(node.out_nodes[0]):
            new_in_nodes = list(consumer.in_nodes)
            new_in_tensors = list(consumer.in_tensors)
            for edge_idx in range(len(new_in_nodes)):
                in_node, in_slot = new_in_nodes[edge_idx]
                if in_node.id != node.id:
                    continue
                if in_slot != 0:
                    raise RuntimeError("[MegaKittens] in_slot != 0 for view op consumer")
                new_in_nodes[edge_idx] = (source, source_slot)
                new_in_tensors[edge_idx] = view_shape
                source.out_nodes[source_slot].append(consumer)
            consumer.in_nodes = tuple(new_in_nodes)
            consumer.in_tensors = tuple(new_in_tensors)

    if not view_node_ids:
        return

    dag.nodes = [n for n in dag.nodes if n.id not in view_node_ids]
    for node in dag.nodes:
        for slot_idx in range(len(node.out_nodes)):
            node.out_nodes[slot_idx][:] = [n for n in node.out_nodes[slot_idx] if n.id not in view_node_ids]

    for node in dag.nodes:
        if node.is_output:
            for edge_idx, ((in_node, in_slot), in_tensor) in enumerate(zip(node.in_nodes, node.in_tensors)):
                src_tensor = in_node.out_tensors[in_slot]
                if in_tensor.numel != src_tensor.numel:
                    raise RuntimeError(
                        f"[MegaKittens] Output node view has incompatible numel: "
                        f"view={in_tensor.shape} (numel={in_tensor.numel}) vs "
                        f"source={src_tensor.shape} (numel={src_tensor.numel})"
                    )


def merge_transpose_ops(dag: DAG) -> None:
    """Merge transpose nodes into downstream ops."""
    transpose_node_ids: set[int] = set()

    for node in dag.nodes:
        if not isinstance(node.itype, str) or not node.itype.startswith("transpose:"):
            continue

        perm_4d = tuple(int(x) for x in node.itype.split(":")[1].split(","))
        if len(perm_4d) != 4:
            raise RuntimeError("[MegaKittens] Permutation isn't 4D")
        inverse_perm = [0] * len(perm_4d)
        for i, p in enumerate(perm_4d):
            inverse_perm[p] = i

        transpose_node_ids.add(node.id)
        source, source_slot = node.in_nodes[0]

        for consumer in node.out_nodes[0]:
            if not isinstance(consumer.itype, Gemm):
                # TODO: support standalone transposes
                raise RuntimeError(
                    f"[MegaKittens] Transpose node feeds into non-Gemm consumer "
                    f"(itype={consumer.itype!r}). Standalone transposes are not supported."
                )

            new_in_nodes = list(consumer.in_nodes)
            new_in_ranges = list(consumer.in_ranges)
            new_in_tensors = list(consumer.in_tensors)

            for edge_idx in range(len(new_in_nodes)):
                in_node, in_slot = new_in_nodes[edge_idx]
                if in_node.id != node.id:
                    continue
                if in_slot != 0:
                    raise RuntimeError(f"[MegaKittens] Transpose node referenced with output slot {in_slot}, expected 0")

                if edge_idx == 0:
                    consumer.itype = Gemm(transpose_a=True, transpose_b=consumer.itype.transpose_b)
                elif edge_idx == 1:
                    consumer.itype = Gemm(transpose_a=consumer.itype.transpose_a, transpose_b=True)
                else:
                    raise RuntimeError("[MegaKittens] Transpose feeds into unexpected Gemm input slot")

                node_out_range = TensorRange.compose(node.out_ranges[0], consumer.in_ranges[edge_idx])
                node_out_range_unpermuted = TensorRange(ranges=tuple(node_out_range.ranges[inverse_perm[i]] for i in range(len(node_out_range.ranges))))
                new_in_nodes[edge_idx] = (source, source_slot)
                new_in_ranges[edge_idx] = TensorRange.compose(node.in_ranges[0], node_out_range_unpermuted)
                new_in_tensors[edge_idx] = source.out_tensors[source_slot]

            if consumer not in source.out_nodes[source_slot]:  # for cases like torch.matmul(x, x.T)
                source.out_nodes[source_slot].append(consumer)
            consumer.in_nodes = tuple(new_in_nodes)
            consumer.in_ranges = tuple(new_in_ranges)
            consumer.in_tensors = tuple(new_in_tensors)

    if not transpose_node_ids:
        return

    dag.nodes = [n for n in dag.nodes if n.id not in transpose_node_ids]
    for node in dag.nodes:
        for slot_idx in range(len(node.out_nodes)):
            node.out_nodes[slot_idx][:] = [n for n in node.out_nodes[slot_idx] if n.id not in transpose_node_ids]


def optimize_dag(dag: DAG) -> DAG:
    """Apply optimization passes to the DAG before scheduling."""
    prune_dead_nodes(dag)
    merge_getitem_select_slice_ops(dag)
    merge_transpose_ops(dag)
    merge_view_ops(dag)

    return dag
