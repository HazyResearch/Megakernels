from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

from .dag import Edge, Node

_GRAPH_DUMP_COUNTER = itertools.count()


def save_dag(
  nodes: List[Node],
  edges: List[Edge],
  fn: Callable[..., Any],
) -> None:
    """
    Save a DAG (nodes, edges) as both JSON and a rendered PNG.
    Generates a path from ``fn``'s qualified name under ``megakittens_graphs/``,
    then writes ``{path}.json`` and ``{path}.png``.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        raise ImportError(
        "Graph export requires 'matplotlib' and 'networkx'."
        "Install them with:\n\n"
        " pip install matplotlib networkx\n"
        )

    suffix = next(_GRAPH_DUMP_COUNTER)
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", fn.__qualname__).strip("._") or "graph"
    base_path = Path.cwd() / "megakittens_graphs" / f"{safe}.{suffix:02d}"

    node_key_by_id = {id(node): f"n{idx}" for idx, node in enumerate(nodes)}
    node_by_key = {f"n{idx}": node for idx, node in enumerate(nodes)}
    node_order = {f"n{idx}": idx for idx in range(len(nodes))}

    layout_graph = nx.DiGraph()
    for node_key in node_by_key:
        layout_graph.add_node(node_key)

    for edge in edges:
        src = node_key_by_id.get(id(edge.in_nodes[0]))
        dst = node_key_by_id.get(id(edge.out_nodes[0]))
        if src is None or dst is None:
        continue
        layout_graph.add_edge(src, dst)

    generations = [
        sorted(generation, key=lambda name: node_order[name])
        for generation in nx.topological_generations(layout_graph)
    ]

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

    # Handle multi-edges visually (layout_graph is a DiGraph, so draw edges manually).
    pair_counts: Dict[tuple[str, str], int] = {}
    pairs: list[tuple[str, str]] = []
    for edge in edges:
        src = node_key_by_id.get(id(edge.in_nodes[0]))
        dst = node_key_by_id.get(id(edge.out_nodes[0]))
        if src is None or dst is None:
        continue
        pair = (src, dst)
        pairs.append(pair)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    pair_seen: Dict[tuple[str, str], int] = {}
    input_ordinal_by_dst: Dict[str, int] = {}
    for edge in edges:
        src = node_key_by_id.get(id(edge.in_nodes[0]))
        dst = node_key_by_id.get(id(edge.out_nodes[0]))
        if src is None or dst is None:
        continue

        pair = (src, dst)
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
        edgelist=[(src, dst)],
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

        ordinal = input_ordinal_by_dst.get(dst, 0)
        input_ordinal_by_dst[dst] = ordinal + 1

        x0, y0 = pos[src]
        x1, y1 = pos[dst]
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

    sinks = {node for node, out_deg in layout_graph.out_degree() if out_deg == 0}

    for node_key, (x, y) in pos.items():
        node = node_by_key[node_key]
        label_parts = [
        node_key,
        str(node.default),
        f"dtype={node.dtype.value}",
        f"shape={tuple(node.shape)}",
        ]

        default_str = str(node.default)
        if default_str.startswith("input["):
        fill = "#dff3df"
        elif default_str.startswith("attr["):
        fill = "#f7efc6"
        elif node_key in sinks:
        fill = "#f8d7da"
        else:
        fill = "#dbeafe"

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

    base_path.parent.mkdir(parents=True, exist_ok=True)

    node_index_by_id = {id(node): idx for idx, node in enumerate(nodes)}
    dag_json = {
        "nodes": [
        {"id": idx, "dtype": node.dtype.value, "shape": list(node.shape), "device": node.device, "default": node.default}
        for idx, node in enumerate(nodes)
        ],
        "edges": [
        {
            "optype": edge.optype.value,
            "src": node_index_by_id.get(id(edge.in_nodes[0])),
            "dst": node_index_by_id.get(id(edge.out_nodes[0])),
        }
        for edge in edges
        ],
    }

    json_path = base_path.with_suffix(".json")
    json_path.write_text(json.dumps(dag_json, indent=2))

    png_path = base_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[MegaKittens] Saved DAG to {base_path}")
