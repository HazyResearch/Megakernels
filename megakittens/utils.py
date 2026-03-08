from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

from .dag import Node

_GRAPH_DUMP_COUNTER = itertools.count()


def make_graph_base_path(fn: Callable[..., Any]) -> Path:
    safe_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", fn.__qualname__).strip("._") or "graph"
    suffix = next(_GRAPH_DUMP_COUNTER)
    return Path.cwd() / "megakittens_graphs" / f"{safe_name}.{suffix:02d}"


def save_json(
    nodes: List[Node],
    base_path: Path,
) -> dict[str, Any]:
    """
    Build a DAG JSON payload from node objects.
    """
    node_index_by_id: Dict[int, int] = {id(node): idx for idx, node in enumerate(nodes)}
    dag_json = {
        "nodes": [
            {
                "id": idx,
                "optype": node.optype.value,
                "input_index": node.input_index,
                "in_nodes": [node_index_by_id[id(in_node)] for in_node in node.in_nodes],
                "out_nodes": [
                    [node_index_by_id[id(out_node)] for out_node in out_nodes]
                    for out_nodes in node.out_nodes
                ],
                "out_tensors": [
                    {
                        "dtype": tensor.dtype.value,
                        "shape": [int(dim) for dim in tensor.shape],
                        "device": tensor.device.model_dump(),
                    }
                    for tensor in node.out_tensors
                ],
            }
            for idx, node in enumerate(nodes)
        ],
    }

    json_path = base_path.with_suffix(".json")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(dag_json, indent=2))

    return dag_json


def save_dag(
    nodes: List[Node],
    base_path: Path,
) -> None:
    """
    Render and save a DAG from node objects.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        raise ImportError(
            "Graph export requires 'matplotlib' and 'networkx'. "
            "Install them with:\n\n"
            "pip install matplotlib networkx\n"
        )

    if not isinstance(nodes, list):
        raise RuntimeError("[MegaKittens] DAG payload is invalid")
    for node in nodes:
        if not isinstance(node, Node):
            raise RuntimeError("[MegaKittens] DAG node entry is invalid")

    ################################
    # Build lookup tables
    ################################
    node_key_by_id: Dict[int, str] = {}
    node_by_key: Dict[str, Node] = {}
    node_order: Dict[str, int] = {}
    for idx, node in enumerate(nodes):
        key = f"N{idx}"
        node_key_by_id[id(node)] = key
        node_by_key[key] = node
        node_order[key] = idx

    ################################
    # Build layout graph from edges
    ################################
    layout_graph = nx.DiGraph()
    for node_key in node_by_key:
        layout_graph.add_node(node_key)

    drawable_edges: list[tuple[Edge, str, str]] = []
    pair_counts: Dict[tuple[str, str], int] = {}
    for edge in edges:
        src_keys = [node_key_by_id.get(id(src_node)) for src_node in edge.in_nodes]
        dst_keys = [node_key_by_id.get(id(dst_node)) for dst_node in edge.out_nodes]
        for src_key in src_keys:
            for dst_key in dst_keys:
                if src_key is None or dst_key is None:
                    raise RuntimeError("[MegaKittens] Key is None during graph export")
                drawable_edges.append((edge, src_key, dst_key))
                layout_graph.add_edge(src_key, dst_key)
                pair = (src_key, dst_key)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    try:
        generation_lists = list(nx.topological_generations(layout_graph))
    except Exception as exc:
        raise RuntimeError("[MegaKittens] Cannot lay out DAG: graph is not acyclic")

    generations = [
        sorted(generation, key=lambda name: node_order.get(name, 0))
        for generation in generation_lists
    ]

    ################################
    # Compute node coordinates
    ################################
    x_spacing = 4.2
    y_spacing = 2.8
    pos: Dict[str, tuple[float, float]] = {}
    for x_index, generation in enumerate(generations):
        offset = (len(generation) - 1) / 2.0
        for y_index, node_key in enumerate(generation):
            pos[node_key] = (x_index * x_spacing, (offset - y_index) * y_spacing)

    fig_width = max(10.0, 4.0 * max(1, len(generations)))
    max_generation_size = max((len(generation) for generation in generations), default=1)
    fig_height = max(6.0, 1.8 * max_generation_size)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)

    ################################
    # Draw edges and edge labels
    ################################
    pair_seen: Dict[tuple[str, str], int] = {}
    input_ordinal_by_dst: Dict[str, int] = {}
    for edge, src_key, dst_key in drawable_edges:
        pair = (src_key, dst_key)
        seen = pair_seen.get(pair, 0)
        pair_seen[pair] = seen + 1

        multiplicity = pair_counts[pair]
        if multiplicity == 1:
            rad = 0.0
        else:
            center = (multiplicity - 1) / 2.0
            rad = 0.18 * (seen - center)

        nx.draw_networkx_edges(
            layout_graph,
            pos,
            edgelist=[(src_key, dst_key)],
            ax=ax,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            width=1.4,
            edge_color="#555555",
            connectionstyle=f"arc3,rad={rad}",
            min_source_margin=18,
            min_target_margin=18,
        )

        ordinal = input_ordinal_by_dst.get(dst_key, 0)
        input_ordinal_by_dst[dst_key] = ordinal + 1

        if src_key not in pos or dst_key not in pos:
            continue
        x0, y0 = pos[src_key]
        x1, y1 = pos[dst_key]
        label_x = 0.5 * (x0 + x1)
        label_y = 0.5 * (y0 + y1) + (0.35 + abs(rad)) * (1 if y0 <= y1 else -1) * (1 if rad >= 0 else -1)

        ax.text(
            label_x,
            label_y,
            f"{edge.optype.value} [{ordinal}]",
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
        )

    ################################
    # Draw nodes
    ################################
    sinks = {node_key for node_key, out_deg in layout_graph.out_degree() if out_deg == 0}

    for node_key, (x, y) in pos.items():
        node = node_by_key[node_key]
        if node.input_index >= 0:
            role = f"input[{node.input_index}]"
            fill = "#dff3df"
        elif node_key in sinks:
            role = "output"
            fill = "#f8d7da"
        elif layout_graph.in_degree(node_key) == 0:
            role = "attr"
            fill = "#f7efc6"
        else:
            role = "op"
            fill = "#dbeafe"

        label_parts = [
            node_key,
            f"{role}",
            f"dtype={node.dtype.value}",
            f"shape={tuple(node.shape)}",
            f"device={node.device}",
        ]

        ax.text(
            x,
            y,
            "\n".join(label_parts),
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": fill, "edgecolor": "black", "linewidth": 1.0},
        )

    ax.set_axis_off()
    fig.tight_layout()

    png_path = base_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[MegaKittens] Saved DAG to {png_path}")
