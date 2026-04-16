from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import Any, Dict, List

import torch
from torch.fx.passes.shape_prop import TensorMetadata

from .schema.dag import DAG, Node
from .schema.itype import IType
from .schema.tensor import TensorMeta


def _resolve_itype(gm: torch.fx.GraphModule, node: torch.fx.Node) -> tuple[IType, list[int]]:
    if node.op == "call_function":
        target = node.target
    elif node.op == "call_method":
        target = node.target
    elif node.op == "call_module":
        try:
            module = gm.get_submodule(node.target)
        except Exception:
            raise RuntimeError(f"[MegaKittens] Invalid call_module node '{node.name}' target={node.target!r}")
        target = type(module)
    else:
        raise RuntimeError(f"[MegaKittens] Unsupported node op '{node.op}' for node '{node.name}'")
    result = IType.from_torch(target, args=node.args, kwargs=node.kwargs)
    if isinstance(result, tuple):
        itype, aten_output_indices = result
        if not isinstance(itype, IType) or not isinstance(aten_output_indices, list):
            raise RuntimeError(f"[MegaKittens] Invalid resolver result for node '{node.name}': {result!r}")
        return itype, aten_output_indices
    if not isinstance(result, IType):
        raise RuntimeError(f"[MegaKittens] Invalid resolve result for node '{node.name}': expected IType, got {type(result).__name__}")
    return result, []


def _get_output_tensors(node: torch.fx.Node, aten_output_indices: list[int] = []) -> tuple[TensorMeta, ...]:
    if "tensor_meta" in node.meta:
        tensor_meta = node.meta["tensor_meta"]
        val = node.meta.get("val")
        if isinstance(tensor_meta, (tuple, list)):
            if all(isinstance(tm, TensorMetadata) for tm in tensor_meta):
                if val is None:
                    return tuple(TensorMeta.from_torch(tm) for tm in tensor_meta)

                elif isinstance(val, torch.Tensor):
                    if len(tensor_meta) != 1:
                        raise RuntimeError(
                            f"[MegaKittens] Node '{node.name}' has {len(tensor_meta)} tensor metadata outputs"
                            f" but single Tensor 'val' provided"
                        )
                    return (TensorMeta.from_torch(tensor_meta[0], fallback_device=val.device),)

                if not isinstance(val, (tuple, list)):
                    raise RuntimeError(
                        f"[MegaKittens] Node '{node.name}' has multiple tensor metadata outputs"
                        f" but 'val' is not a tuple/list"
                    )
                if any(not isinstance(v, torch.Tensor) for v in val):
                    raise RuntimeError(
                        f"[MegaKittens] Node '{node.name}' has multiple tensor metadata outputs"
                        f" but 'val' contains non-tensor entries"
                    )
                if len(val) != len(tensor_meta):
                    raise RuntimeError(
                        f"[MegaKittens] Node '{node.name}' has {len(tensor_meta)} tensor metadata outputs"
                        f" but 'val' has {len(val)} elements"
                    )

                return tuple(
                    TensorMeta.from_torch(tm, fallback_device=val[idx].device)
                    for idx, tm in enumerate(tensor_meta)
                )

            elif isinstance(val, torch.Tensor):
                return (TensorMeta.from_torch(val),)

            else:
                raise RuntimeError(
                    f"[MegaKittens] Node '{node.name}' has tuple/list tensor metadata entries that are not TensorMetadata"
                    f" and 'val' is not a Tensor (type={type(val).__name__})"
                )

        elif isinstance(tensor_meta, TensorMetadata):
            if val is not None and not isinstance(val, torch.Tensor):
                raise RuntimeError(
                    f"[MegaKittens] Node '{node.name}' has single tensor metadata output"
                    f" but 'val' is not a Tensor (type={type(val).__name__})"
                )
            fallback_device = val.device if isinstance(val, torch.Tensor) else None
            return (TensorMeta.from_torch(tensor_meta, fallback_device=fallback_device),)
        raise RuntimeError(
            f"[MegaKittens] Unsupported tensor_meta type for node '{node.name}'"
            f" (type={type(tensor_meta).__name__})"
        )

    elif "val" in node.meta:
        value = node.meta["val"]
        if isinstance(value, (tuple, list)):
            tensors = []
            if aten_output_indices:
                for i in aten_output_indices:
                    if i >= len(value):
                        raise RuntimeError(
                            f"[MegaKittens] aten_output_indices[{i}] out of range for node '{node.name}'"
                            f" (val has {len(value)} entries)"
                        )
                    if not isinstance(value[i], torch.Tensor):
                        raise RuntimeError(
                            f"[MegaKittens] Non-tensor entry at index {i} in node metadata for node '{node.name}'"
                            f" (type={type(value[i]).__name__})"
                        )
                    tensors.append(TensorMeta.from_torch(value[i]))
                return tuple(tensors)
            else:
                for v in value:
                    if not isinstance(v, torch.Tensor):
                        raise RuntimeError(
                            f"[MegaKittens] Non-tensor entry in node metadata for node '{node.name}'"
                            f" (indexed entry type={type(v).__name__})"
                        )
                    tensors.append(TensorMeta.from_torch(v))
            return tuple(tensors)
        elif isinstance(value, torch.Tensor):
            return (TensorMeta.from_torch(value),)
        else:
            raise RuntimeError(
                f"[MegaKittens] Node metadata is not a Tensor for node '{node.name}'"
                f" (type={type(value).__name__})"
            )

    else:
        raise RuntimeError(
            f"[MegaKittens] Missing tensor metadata for node '{node.name}'"
            f" (op={node.op}, target={node.target!r}, meta_keys={list(node.meta.keys())})"
        )


def _resolve_input_node_and_output_index(
    input_node: torch.fx.Node,
) -> tuple[torch.fx.Node, int]:
    output_idx = 0
    current = input_node
    while current.op == "call_function" and current.target == operator.getitem:
        args = current.args
        if len(args) < 2:
            raise RuntimeError(
                f"[MegaKittens] Unsupported getitem usage for node '{current.name}'"
                f" (op={current.op}, target={current.target!r}, args={args!r})"
            )
        root = args[0]
        if not isinstance(root, torch.fx.Node):
            raise RuntimeError(
                f"[MegaKittens] Cannot resolve FX getitem input for node '{current.name}'"
                f" because source is not an FX node (type={type(root).__name__})"
            )
        item = args[1]
        if not isinstance(item, int):
            raise RuntimeError(f"[MegaKittens] Unsupported getitem index type for node '{current.name}'")
        output_idx = item
        current = root
    if not isinstance(current, torch.fx.Node):
        raise RuntimeError(
            f"[MegaKittens] Failed to resolve input node for '{input_node.name}'"
        )
    return current, output_idx


def _flatten_output_nodes(
    value: Any,
) -> list[torch.fx.Node]:
    if isinstance(value, torch.fx.Node):
        return [value]
    elif isinstance(value, (list, tuple)):
        nodes: list[torch.fx.Node] = []
        for item in value:
            nodes.extend(_flatten_output_nodes(item))
        return nodes
    elif isinstance(value, dict):
        nodes: list[torch.fx.Node] = []
        for item in value.values():
            nodes.extend(_flatten_output_nodes(item))
        return nodes
    else:
        raise RuntimeError(f"[MegaKittens] Unsupported output value type '{type(value).__name__}'")


def prune_fx_graph(fx_nodes: Iterable[torch.fx.Node]) -> list[torch.fx.Node]:
    """Given an iterable of Torch FX graph nodes, preserve only the nodes that feed to the output."""
    fx_nodes = list(fx_nodes)  # materialize so we can iterate twice
    output_nodes = [node for node in fx_nodes if node.op == "output"]
    if len(output_nodes) != 1:  # FX graph always produces 1 output node
        raise RuntimeError(f"[MegaKittens] Number of output nodes is {len(output_nodes)}")
    output_node = output_nodes[0]
    # TODO: support void functions
    if not output_node.args or len(output_node.args) != 1 or output_node.args[0] is None:
        raise RuntimeError("[MegaKittens] Void output graphs are not supported. Please return at least one tensor.")
    valid_names: set[str] = set()
    node_stack: List[torch.fx.Node] = [output_node]
    while node_stack:
        current = node_stack.pop()
        if current.name in valid_names:  # already visited
            continue
        valid_names.add(current.name)
        node_stack.extend(current.all_input_nodes)
    return [
        node for node in fx_nodes
        if node.name in valid_names
        and not (node.op == "call_function" and node.target == operator.getitem)
    ]

def trace(gm: torch.fx.GraphModule, example_inputs: List[Any]) -> DAG:
    """
    Convert an FX GraphModule plus example inputs into a MegaKittens node-centric DAG.

    Returns:
        A list of `dag.Node` values with topology represented by node connectivity fields.

    Dead nodes (not reachable from the output) are pruned.
    """

    # Prune to only nodes that feed into the output.
    fx_nodes = prune_fx_graph(gm.graph.nodes)

    # Maps FX node name -> {aten_output_idx: itype_output_idx} for multi-output ops
    output_idx_remap: Dict[str, Dict[int, int]] = {}

    # Extract all DAG nodes
    _input_index: int = 0
    node_by_name: Dict[str, Node] = {}
    dag_nodes: List[Node] = []

    for node in fx_nodes:
        is_input = False
        is_output = False
        itype: IType | None = None
        input_index: int | None = None
        out_tensors: tuple[TensorMeta, ...] | None = None

        if node.op == "placeholder":
            is_input = True
            input_nodes = list(node.all_input_nodes)
            if _input_index >= len(example_inputs):
                raise RuntimeError("[MegaKittens] Number of input nodes is greater than len(example_inputs)")
            input_index = _input_index
            example_input = example_inputs[_input_index]
            _input_index += 1
            if isinstance(example_input, torch.Tensor):
                out_tensors = (TensorMeta.from_torch(example_input),)
            else:
                raise RuntimeError("[MegaKittens] Non-tensor inputs are not supported")

        elif node.op == "get_attr":
            is_input = True
            input_nodes = list(node.all_input_nodes)
            try:
                attr = getattr(gm, node.target)
            except Exception:
                raise RuntimeError("[MegaKittens] Invalid get_attr node")
            if isinstance(attr, torch.Tensor):
                out_tensors = (TensorMeta.from_torch(attr),)
            else:
                raise RuntimeError("[MegaKittens] Non-tensor attributes are not supported")

        elif node.op in {"call_function", "call_module", "call_method"}:
            itype, aten_output_indices = _resolve_itype(gm, node)
            input_nodes = list(node.all_input_nodes)
            out_tensors = _get_output_tensors(node, aten_output_indices=aten_output_indices)
            if aten_output_indices:
                output_idx_remap[node.name] = {
                    aten_idx: itype_idx for itype_idx, aten_idx in enumerate(aten_output_indices)
                }

        elif node.op == "output":
            is_output = True
            if not node.args:
                raise RuntimeError("[MegaKittens] Output node has no args")
            input_nodes = _flatten_output_nodes(node.args)
            if "tensor_meta" in node.meta or "val" in node.meta:
                out_tensors = _get_output_tensors(node)

        else:
            raise RuntimeError(f"[MegaKittens] Invalid node op {node.op}")

        in_nodes_list = []
        for input_node in input_nodes:
            if input_node.op == "output":
                raise RuntimeError(
                    f"[MegaKittens] Invalid input graph edge: node '{node.name}' consumes output node '{input_node.name}'"
                )
            resolved_node, output_idx = _resolve_input_node_and_output_index(input_node)
            if resolved_node.name in output_idx_remap:
                output_idx = output_idx_remap[resolved_node.name][output_idx]
            if resolved_node.name not in node_by_name:
                raise RuntimeError(f"[MegaKittens] No source node '{resolved_node.name}' exists for '{node.name}'")
            source_node = node_by_name[resolved_node.name]
            if output_idx >= len(source_node.out_tensors):
                raise RuntimeError(
                    f"[MegaKittens] Node '{node.name}' reads output slot {output_idx} of '{resolved_node.name}',"
                    f" but source node has {len(source_node.out_tensors)} outputs"
                )
            in_nodes_list.append((source_node, output_idx))

        if node.op == "output" and out_tensors is None:
            if not in_nodes_list:
                raise RuntimeError(f"[MegaKittens] Output node '{node.name}' has no resolvable inputs")
            out_tensors = tuple(
                in_node.out_tensors[output_idx] for in_node, output_idx in in_nodes_list
            )

        current_node = Node(
            is_input=is_input,
            is_output=is_output,
            itype=itype,
            in_nodes=tuple(in_nodes_list),
            in_ranges=tuple(src_node.out_tensors[slot].full_range for src_node, slot in in_nodes_list),
            out_tensors=out_tensors,
            out_ranges=tuple(tensor_meta.full_range for tensor_meta in out_tensors),
            out_nodes=tuple([] for _ in out_tensors),
            input_index=input_index,
        )

        node_by_name[node.name] = current_node
        dag_nodes.append(current_node)

        # Add this node to each source node's outgoing tuple group.
        for input_node, output_idx in in_nodes_list:
            if output_idx >= len(input_node.out_nodes):
                raise RuntimeError(
                    f"[MegaKittens] Output slot {output_idx} missing for source node '{input_node}'"
                )
            input_node.out_nodes[output_idx].append(current_node)

    if _input_index != len(example_inputs):
        raise RuntimeError(
            f"[MegaKittens] Number of input nodes is {_input_index}, but len(example_inputs) is {len(example_inputs)}"
        )

    return DAG(dag_nodes)
