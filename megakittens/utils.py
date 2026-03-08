from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch

from .dag import Node
from .instruction import Instruction

_LOG_DUMP_COUNTER = itertools.count()


def create_log_base_path(fn: Callable[..., Any]) -> Path:
    safe_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", fn.__qualname__).strip("._")
    if not safe_name:
        raise ValueError(
            f"[MegaKittens] Unable to construct a valid log file base name for function {fn!r}"
        )
    suffix = next(_LOG_DUMP_COUNTER)
    return Path.cwd() / "log" / f"{safe_name}.{suffix:02d}"


def save_dag_as_png_as_json(
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
                "in_nodes": [
                    [node_index_by_id[id(in_node)], input_slot]
                    for in_node, input_slot in node.in_nodes
                ],
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

    json_path = base_path.parent / (base_path.name + ".graph.json")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(dag_json, indent=2))

    return dag_json


def save_dag_as_png(
    dag_json: dict[str, Any],
    base_path: Path,
) -> None:
    """
    Render a DAG JSON payload (from save_dag_as_png_as_json) as a PNG using graphviz.
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            "[MegaKittens] `graphviz` package is required for DAG export. "
            "Install it with: pip install graphviz"
        )

    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="TB")
    dot.attr("node", shape="record", style="filled", fillcolor="#e8e8e8", fontname="Menlo")
    dot.attr("edge", fontname="Menlo")

    for node in dag_json["nodes"]:
        nid = str(node["id"])
        optype = node["optype"]
        op_line = f"Input[{node['input_index']}]" if optype == "input" else optype.capitalize()
        lines = [f"#{nid}", op_line]
        for t in node["out_tensors"]:
            shape_str = "\u00d7".join(str(d) for d in t["shape"])
            dev = t["device"]
            device_str = f"{dev['type']}:{dev['index']}" if dev["index"] is not None else dev["type"]
            lines.append(f"- {t['dtype'].upper()} [{shape_str}] {device_str.upper()}")
        label = "\\n".join(lines)

        if optype == "input":
            dot.node(nid, label=label, fillcolor="#a8d8a8")
        elif optype == "output":
            dot.node(nid, label=label, fillcolor="#d8a8a8")
        else:
            dot.node(nid, label=label, fillcolor="#a8c8d8")

    for node in dag_json["nodes"]:
        for src_id, input_slot in node["in_nodes"]:
            dot.edge(str(src_id), str(node["id"]), label=f" {input_slot} ")

    base_path.parent.mkdir(parents=True, exist_ok=True)
    dot.render(filename=str(base_path) + ".graph", cleanup=True)


def save_schedule_as_txt(
    tensors: List[torch.Tensor],
    instructions: List[Instruction],
    base_path: Path,
) -> None:
    """
    Write a human-readable schedule dump to a .txt file.
    """
    lines: List[str] = []

    lines.append(f"Tensors: {len(tensors)}")
    lines.append("-" * 60)
    for idx, t in enumerate(tensors):
        # TODO: print barrier tensor
        shape_str = "x".join(str(d) for d in t.shape)
        lines.append(f"  T{idx:<4d}  dtype={t.dtype}  shape=[{shape_str}]  device={t.device}")

    lines.append("")
    lines.append(f"Instructions: {len(instructions)}")
    lines.append("-" * 60)
    # TODO: dynamically column-align all fields
    itype_width = max((len(inst.itype.name) for inst in instructions), default=0)
    itype_width = max(itype_width, 12)
    for idx, inst in enumerate(instructions):
        src_field = f"src={[f'T{t}' for t in inst.src_tensors]}"
        dst_field = f"dst={[f'T{t}' for t in inst.dst_tensors]}"
        idx_field = f"idx={list(inst.indices)}"
        src_bar_field = f"src_bar={list(inst.src_barriers)}"
        src_bar_tgt_field = f"src_bar_tgt={list(inst.src_barrier_targets)}"
        dst_bar_field = f"dst_bar={list(inst.dst_barrier)}"
        lines.append(
            f"  I{idx:<5d} {inst.itype.name:<{itype_width}}"
            f"  {src_field:<16}"
            f" {dst_field:<16}"
            f" {idx_field:<20}"
            f" {src_bar_field:<16}"
            f" {src_bar_tgt_field:<28}"
            f" {dst_bar_field}"
        )

    txt_path = base_path.parent / (base_path.name + ".schedule.txt")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines) + "\n")
