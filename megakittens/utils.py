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
    num_barriers: int,
    base_path: Path,
) -> None:
    """
    Write a human-readable schedule dump to a .txt file.
    """
    lines: List[str] = []

    lines.append(f"Tensors: {len(tensors)}")
    lines.append(f"Barriers: {num_barriers}")
    lines.append(f"Instructions: {len(instructions)}")
    lines.append("-" * 60)

    if tensors:
        t_id_strs = [f"T{idx}" for idx in range(len(tensors))]
        t_dtype_strs = [f"dtype={t.dtype}" for t in tensors]
        t_shape_strs = [f"shape=[{'x'.join(str(d) for d in t.shape)}]" for t in tensors]
        t_device_strs = [f"device={t.device}" for t in tensors]

        w_t_id = max(len(s) for s in t_id_strs)
        w_t_dtype = max(len(s) for s in t_dtype_strs)
        w_t_shape = max(len(s) for s in t_shape_strs)

        for i in range(len(tensors)):
            lines.append(
                f"  {t_id_strs[i]:<{w_t_id}}"
                f"  {t_dtype_strs[i]:<{w_t_dtype}}"
                f"  {t_shape_strs[i]:<{w_t_shape}}"
                f"  {t_device_strs[i]}"
            )

    lines.append("")
    lines.append("-" * 60)

    if instructions:
        id_strs = [f"I{idx}" for idx in range(len(instructions))]
        itype_strs = [inst.itype.name for inst in instructions]
        src_strs = [f"src=[{', '.join(f'T{t}' for t in inst.src_tensors)}]" for inst in instructions]
        dst_strs = [f"dst=[{', '.join(f'T{t}' for t in inst.dst_tensors)}]" for inst in instructions]
        idx_strs = [f"idx={list(inst.indices)}" for inst in instructions]
        src_bar_strs = [f"src_bar={list(inst.src_barriers)}" for inst in instructions]
        src_bar_tgt_strs = [f"src_bar_tgt={list(inst.src_barrier_targets)}" for inst in instructions]
        dst_bar_strs = [f"dst_bar={list(inst.dst_barrier)}" for inst in instructions]

        w_id = max(len(s) for s in id_strs)
        w_itype = max(len(s) for s in itype_strs)
        w_src = max(len(s) for s in src_strs)
        w_dst = max(len(s) for s in dst_strs)
        w_idx = max(len(s) for s in idx_strs)
        w_src_bar = max(len(s) for s in src_bar_strs)
        w_src_bar_tgt = max(len(s) for s in src_bar_tgt_strs)

        for i in range(len(instructions)):
            lines.append(
                f"  {id_strs[i]:<{w_id}}"
                f"  {itype_strs[i]:<{w_itype}}"
                f"  {src_strs[i]:<{w_src}}"
                f"  {dst_strs[i]:<{w_dst}}"
                f"  {idx_strs[i]:<{w_idx}}"
                f"  {src_bar_strs[i]:<{w_src_bar}}"
                f"  {src_bar_tgt_strs[i]:<{w_src_bar_tgt}}"
                f"  {dst_bar_strs[i]}"
            )

    txt_path = base_path.parent / (base_path.name + ".schedule.txt")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines) + "\n")
