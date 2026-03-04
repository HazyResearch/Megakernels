from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping

import torch
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd


def _normalize_fx_value(value: Any) -> Any:
    if isinstance(value, torch.fx.Node):
        return {"kind": "node_ref", "name": value.name}

    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
            "stride": list(value.stride()),
        }

    if isinstance(value, torch.Size):
        return {"kind": "size", "value": list(value)}

    if isinstance(value, slice):
        return {
            "kind": "slice",
            "start": _normalize_fx_value(value.start),
            "stop": _normalize_fx_value(value.stop),
            "step": _normalize_fx_value(value.step),
        }

    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_normalize_fx_value(item) for item in value],
        }

    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_normalize_fx_value(item) for item in value],
        }

    if isinstance(value, Mapping):
        return {
            "kind": "dict",
            "items": {str(key): _normalize_fx_value(item) for key, item in value.items()},
        }

    if isinstance(
        value,
        (
            torch.dtype,
            torch.device,
            torch.layout,
            torch.memory_format,
        ),
    ):
        return str(value)

    if isinstance(value, (bool, int, float, str)) or value is None:
        return value

    return repr(value)


def _fx_graph_to_dag_dict(
    gm: torch.fx.GraphModule,
    example_inputs: List[Any],
) -> Dict[str, Any]:
    """
    Convert an FX GraphModule plus example inputs into a pure-Python DAG.

    The returned object is a Python dictionary that represents the DAG. Nodes
    represent values. Directed edges represent data dependencies and are labeled
    by the operation that produces the destination value.

    Dead nodes (not reachable from the output) are pruned from the result.
    """
    dag_nodes = []
    dag_edges = []

    all_graph_nodes = list(gm.graph.nodes)

    # Prune to only nodes that feed into the output.
    output_node = next((n for n in all_graph_nodes if n.op == "output"), None)
    if output_node is None:
        relevant_names = {n.name for n in all_graph_nodes}
    else:
        relevant_names: set[str] = set()
        stack: list[torch.fx.Node] = [output_node]
        while stack:
            current = stack.pop()
            if current.name in relevant_names:
                continue
            relevant_names.add(current.name)
            stack.extend(current.all_input_nodes)

    example_input_by_placeholder: Dict[str, tuple[int, Any | None]] = {}
    placeholder_nodes = [node for node in all_graph_nodes if node.op == "placeholder"]
    for input_index, node in enumerate(placeholder_nodes):
        example_input = example_inputs[input_index] if input_index < len(example_inputs) else None
        example_input_by_placeholder[node.name] = (input_index, example_input)

    graph_nodes = [node for node in all_graph_nodes if node.name in relevant_names]

    serialized_nodes: list[Dict[str, Any]] = []
    serialized_edges: list[Dict[str, Any]] = []
    serialized_example_inputs: list[Dict[str, Any]] = []

    for node in graph_nodes:
        input_index, example_input = example_input_by_placeholder.get(node.name, (None, None))

        if "val" in node.meta:
            value_metadata = _normalize_fx_value(node.meta["val"])
        elif "tensor_meta" in node.meta:
            tensor_meta = node.meta["tensor_meta"]
            device = None
            if isinstance(example_input, torch.Tensor):
                device = str(example_input.device)
            value_metadata = {
                "kind": "tensor",
                "shape": list(tensor_meta.shape),
                "dtype": str(tensor_meta.dtype),
                "device": device,
                "requires_grad": bool(tensor_meta.requires_grad),
                "stride": list(tensor_meta.stride),
            }
        elif example_input is not None:
            value_metadata = _normalize_fx_value(example_input)
        else:
            value_metadata = None

        if node.op == "placeholder" and input_index is not None:
            serialized_example_inputs.append(
                {
                    "index": input_index,
                    "placeholder": node.name,
                    "value": _normalize_fx_value(example_input),
                }
            )

        serialized_nodes.append(
            {
                "name": node.name,
                "op": node.op,
                "target": str(node.target),
                "args": _normalize_fx_value(node.args),
                "kwargs": _normalize_fx_value(node.kwargs),
                "all_input_nodes": [input_node.name for input_node in node.all_input_nodes],
                "users": [user.name for user in node.users if user.name in relevant_names],
                "example_input_index": input_index if node.op == "placeholder" else None,
                "value": value_metadata,
                "meta_keys": sorted(str(key) for key in node.meta.keys()),
            }
        )

    for node in graph_nodes:
        if node.op == "placeholder":
            continue

        edge_op = "output" if node.op == "output" else str(node.target)
        for input_ordinal, input_node in enumerate(node.all_input_nodes):
            if input_node.name not in relevant_names:
                continue
            serialized_edges.append(
                {
                    "src": input_node.name,
                    "dst": node.name,
                    "op": edge_op,
                    "input_ordinal": input_ordinal,
                }
            )

    output_node = next((node.name for node in graph_nodes if node.op == "output"), None)

    return {
        "nodes": serialized_nodes,
        "edges": serialized_edges,
        "placeholder_order": [node.name for node in graph_nodes if node.op == "placeholder"],
        "output_node": output_node,
        "example_inputs": serialized_example_inputs,
    }


def megakittens_backend(
    fn: Callable[..., Any],
    *,
    verify: bool = False,
    profile: bool = False,
    debug: bool = False,
    save_dag: bool = False,
) -> Callable[[torch.fx.GraphModule, List[Any]], Callable[..., Any]]:
    def _megakittens_backend(gm: torch.fx.GraphModule, example_inputs: List[Any]) -> Callable[..., Any]:
        if debug:
            print(f"[MegaKittens] Compiling function `{fn.__qualname__}`")

            print(f"[MegaKittens] FX graph:")
            gm.graph.print_tabular()

            if save_dag:
                from .utils import save_dag as _save_dag
                dag = _fx_graph_to_dag_dict(gm, example_inputs)
                _save_dag(dag, fn=fn)

        return make_boxed_func(gm.forward)

    return aot_autograd(
        fw_compiler=_megakittens_backend,
        bw_compiler=_megakittens_backend,
    )
