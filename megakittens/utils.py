from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

_GRAPH_DUMP_COUNTER = itertools.count()


def save_dag(
    dag: Dict[str, Any],
    fn: Callable[..., Any],
) -> None:
    """
    Save a DAG dict as both JSON and a rendered PNG.

    Generates a path from ``fn``'s qualified name under ``megakittens_graphs/``,
    then writes ``{path}.json`` (the raw DAG) and ``{path}.png``
    (a dataflow diagram where nodes are values and edges are data
    dependencies labeled by the producing operation).
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        raise ImportError(
            "Graph export requires 'matplotlib' and 'networkx'."
            "Install them with:\n\n"
            "  pip install matplotlib networkx\n"
        )

    suffix = next(_GRAPH_DUMP_COUNTER)
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", fn.__qualname__).strip("._") or "graph"
    base_path = Path.cwd() / "megakittens_graphs" / f"{safe}.{suffix:02d}"

    node_by_name = {node["name"]: node for node in dag["nodes"]}

    layout_graph = nx.DiGraph()
    for node_name in node_by_name:
        layout_graph.add_node(node_name)
    for edge in dag["edges"]:
        layout_graph.add_edge(edge["src"], edge["dst"])

    node_order = {node["name"]: index for index, node in enumerate(dag["nodes"])}
    generations = [
        sorted(generation, key=lambda name: node_order[name])
        for generation in nx.topological_generations(layout_graph)
    ]

    x_spacing = 4.2
    y_spacing = 2.8
    pos: Dict[str, tuple[float, float]] = {}
    for x_index, generation in enumerate(generations):
        offset = (len(generation) - 1) / 2.0
        for y_index, node_name in enumerate(generation):
            pos[node_name] = (x_index * x_spacing, (offset - y_index) * y_spacing)

    fig_width = max(10.0, 4.0 * max(1, len(generations)))
    max_generation_size = max((len(generation) for generation in generations), default=1)
    fig_height = max(6.0, 1.8 * max_generation_size)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)

    pair_counts: Dict[tuple[str, str], int] = {}
    for edge in dag["edges"]:
        pair = (edge["src"], edge["dst"])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    pair_seen: Dict[tuple[str, str], int] = {}
    for edge in dag["edges"]:
        src = edge["src"]
        dst = edge["dst"]
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

        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        label_x = 0.5 * (x0 + x1)
        label_y = 0.5 * (y0 + y1) + (0.35 + abs(rad)) * (1 if y0 <= y1 else -1) * (1 if rad >= 0 else -1)
        ax.text(
            label_x,
            label_y,
            f"{edge['op']} [{edge['input_ordinal']}]",
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
        )

    fill_by_op = {
        "placeholder": "#dff3df",
        "output": "#f8d7da",
        "get_attr": "#f7efc6",
    }
    default_fill = "#dbeafe"

    for node_name, (x, y) in pos.items():
        node_data = node_by_name[node_name]

        label_parts = [str(node_data["name"]), f"[{node_data['op']}]"]
        if node_data["op"] == "placeholder" and node_data.get("example_input_index") is not None:
            label_parts.append(f"input[{node_data['example_input_index']}]")
        elif node_data["op"] != "output":
            label_parts.append(str(node_data["target"]))
        _value = node_data.get("value")
        if _value is None:
            value_text = None
        elif isinstance(_value, Mapping) and _value.get("kind") == "tensor":
            shape = tuple(_value.get("shape", []))
            dtype = str(_value.get("dtype", ""))
            dtype = dtype.replace("torch.", "")
            device = _value.get("device")
            fmt_parts = [f"shape={shape}", f"dtype={dtype}"]
            if device is not None:
                fmt_parts.append(f"device={device}")
            value_text = "\n".join(fmt_parts)
        elif isinstance(_value, Mapping) and _value.get("kind") in {"tuple", "list"}:
            item_count = len(_value.get("items", []))
            value_text = f"{_value['kind']}[{item_count}]"
        elif isinstance(_value, Mapping) and _value.get("kind") == "dict":
            item_count = len(_value.get("items", {}))
            value_text = f"dict[{item_count}]"
        else:
            _text = str(_value)
            value_text = _text if len(_text) <= 96 else _text[:93] + "..."

        if value_text:
            label_parts.append(value_text)

        ax.text(
            x,
            y,
            "\n".join(label_parts),
            ha="center",
            va="center",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": fill_by_op.get(str(node_data["op"]), default_fill),
                "edgecolor": "black",
                "linewidth": 1.0,
            },
        )

    ax.set_axis_off()
    fig.tight_layout()

    base_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = base_path.with_suffix(".json")
    json_path.write_text(json.dumps(dag, indent=2))

    png_path = base_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[MegaKittens] Saved DAG to {base_path}")
