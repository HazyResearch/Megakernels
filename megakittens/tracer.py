from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import Any, Dict, List

import torch
from torch.fx.passes.shape_prop import TensorMetadata

from .schema.dag import DAG, Node
from .schema.itype import IType
from .schema.tensor import TensorMeta, TensorRange, TensorSlice


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
        and not (node.op == "call_function" and node.target in {operator.getitem, torch.ops.aten.select.int, torch.ops.aten.slice.Tensor})
    ]


def convert_fx_node_to_itype(gm: torch.fx.GraphModule, node: torch.fx.Node) -> tuple[IType, list[int]]:
    if node.op == "call_function":
        target = node.target
    elif node.op == "call_method":
        target = node.target
    elif node.op == "call_module":
        try:
            target = type(gm.get_submodule(node.target))
        except Exception:
            raise RuntimeError(f"[MegaKittens] Invalid call_module node '{node.name}' target={node.target!r}")
    else:
        raise RuntimeError(f"[MegaKittens] Node op '{node.op}' for node '{node.name}' cannot be converted to an IType")

    result = IType.from_torch(target, args=node.args, kwargs=node.kwargs)

    if isinstance(result, tuple):
        itype, aten_output_indices = result
        if not isinstance(itype, IType) or not isinstance(aten_output_indices, list):
            raise RuntimeError(f"[MegaKittens] Invalid resolver result for node '{node.name}': {result!r}")
        return itype, aten_output_indices
    else:
        if not isinstance(result, IType):
            raise RuntimeError(f"[MegaKittens] Invalid resolve result for node '{node.name}': expected IType, got {type(result).__name__}")
        return result, []


def extract_fx_node_outputs(node: torch.fx.Node, aten_output_indices: list[int] = []) -> tuple[TensorMeta, ...]:
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

                else:
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

        else:
            raise RuntimeError(
                f"[MegaKittens] Unsupported tensor_meta type for node '{node.name}'"
                f" (type={type(tensor_meta).__name__})"
            )

    elif "val" in node.meta:
        value = node.meta["val"]
        if isinstance(value, (tuple, list)):
            if aten_output_indices:
                tensors = []
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
                tensors = []
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


def flatten_fx_nodes(
    val: Any,
) -> list[torch.fx.Node]:
    if isinstance(val, torch.fx.Node):
        return [val]
    elif isinstance(val, (list, tuple)):
        nodes: list[torch.fx.Node] = []
        for item in val:
            nodes.extend(flatten_fx_nodes(item))
        return nodes
    elif isinstance(val, dict):
        nodes: list[torch.fx.Node] = []
        for item in val.values():
            nodes.extend(flatten_fx_nodes(item))
        return nodes
    else:
        raise RuntimeError(f"[MegaKittens] Unsupported node type '{type(val).__name__}'")


def resolve_input_node(input_node: torch.fx.Node) -> tuple[torch.fx.Node, int, list[TensorSlice]]:
    """Walk back through getitem, indexing, and slicing ops to find the real source FX node.

    Returns (source_fx_node, output_idx, slice_chain) where:
      - source_fx_node: first non-view, non-getitem FX node.
      - output_idx: output slot of that source (for multi-output ops).
      - slice_chain: slicing ops in parent-to-leaf order
    """
    output_idx = 0
    slice_chain: list[TensorSlice] = []
    current_node = input_node
    while True:
        if current_node.op == "call_function" and current_node.target == operator.getitem:
            args = current_node.args
            if len(args) < 2:
                raise RuntimeError(
                    f"[MegaKittens] Unsupported getitem usage for node '{current_node.name}'"
                    f" (op={current_node.op}, target={current_node.target!r}, args={args!r})"
                )
            if not isinstance(args[0], torch.fx.Node): # TODO?
                raise RuntimeError(
                    f"[MegaKittens] Cannot resolve FX getitem input for node '{current_node.name}'"
                    f" because source is not an FX node (type={type(args[0]).__name__})"
                )
            if not isinstance(args[1], int):
                raise RuntimeError(f"[MegaKittens] Unsupported getitem index type for node '{current_node.name}'")
            output_idx = args[1]
            current_node = args[0]
        elif current_node.op == "call_function" and current_node.target == torch.ops.aten.select.int:
            args = current_node.args
            if len(args) != 3:
                raise RuntimeError(f"[MegaKittens] aten.select.int expects 3 args for node '{current_node.name}', got {len(args)}")
            if not isinstance(args[0], torch.fx.Node):
                raise RuntimeError(f"[MegaKittens] aten.select.int args[0] is not an FX node for '{current_node.name}'")
            if not isinstance(args[1], int) or not isinstance(args[2], int):
                raise RuntimeError(
                    f"[MegaKittens] aten.select.int requires int dim/idx for node '{current_node.name}',"
                    f" got dim={args[1]!r} idx={args[2]!r}"
                )
            slice_chain.append(TensorSlice(op="select", dim=args[1], start=args[2], end=args[2] + 1))
            current_node = args[0]
        elif current_node.op == "call_function" and current_node.target == torch.ops.aten.slice.Tensor:
            args = current_node.args
            if not (2 <= len(args) <= 5):
                raise RuntimeError(f"[MegaKittens] aten.slice.Tensor expects 2-5 args for node '{current_node.name}', got {len(args)}")
            if not isinstance(args[0], torch.fx.Node):
                raise RuntimeError(f"[MegaKittens] aten.slice.Tensor parent is not an FX node for '{current_node.name}'")
            dim = args[1]
            start = args[2] if len(args) > 2 else None
            end = args[3] if len(args) > 3 else None
            step = args[4] if len(args) > 4 and args[4] is not None else 1
            if not isinstance(dim, int):
                raise RuntimeError(
                    f"[MegaKittens] aten.slice.Tensor requires int dim for node '{current_node.name}',"
                    f" got dim={dim!r}"
                )
            if start is not None and not isinstance(start, int):
                raise RuntimeError(
                    f"[MegaKittens] aten.slice.Tensor start must be int or None for node '{current_node.name}',"
                    f" got start={start!r}"
                )
            if end is not None and not isinstance(end, int):
                raise RuntimeError(
                    f"[MegaKittens] aten.slice.Tensor end must be int or None for node '{current_node.name}',"
                    f" got end={end!r}"
                )
            if step != 1:
                raise RuntimeError(
                    f"[MegaKittens] aten.slice.Tensor step != 1 is not supported for node '{current_node.name}'"
                )
            slice_chain.append(TensorSlice(op="slice", dim=dim, start=start, end=end))
            current_node = args[0]
        else:
            break

    if not isinstance(current_node, torch.fx.Node):
        raise RuntimeError(f"[MegaKittens] Failed to resolve input node for '{input_node.name}'")
    return current_node, output_idx, slice_chain.reverse()  # source-to-destination order


def trace(gm: torch.fx.GraphModule, example_inputs: List[Any]) -> DAG:
    """
    Convert an FX GraphModule plus example inputs into a MegaKittens node-centric DAG.

    Returns:
        A list of `dag.Node` values with topology represented by node connectivity fields.

    Dead nodes (not reachable from the output) are pruned.
    """

    # Prune to only nodes that feed into the output.
    fx_nodes = prune_fx_graph(gm.graph.nodes)

    # Extract all DAG nodes
    input_counter: int = 0
    output_idx_map: Dict[str, Dict[int, int]] = {}  # FX node name -> {aten_output_idx: itype_output_idx}
    node_by_name: Dict[str, Node] = {}
    dag_nodes: List[Node] = []

    for fx_node in fx_nodes:
        is_input = False
        is_output = False
        itype: IType | None = None
        input_index: int | None = None
        input_fx_nodes: Iterable[torch.fx.Node] | None = None
        out_tensors: tuple[TensorMeta, ...] | None = None

        if fx_node.op == "placeholder":
            is_input = True
            input_fx_nodes = ()  # in_nodes are never used for input nodes
            if len(fx_node.all_input_nodes) != 0:
                raise RuntimeError("[MegaKittens] Input node has source inputs.")
            if input_counter >= len(example_inputs):
                raise RuntimeError("[MegaKittens] Number of input nodes is greater than len(example_inputs)")
            input_index = input_counter
            input_counter += 1
            if not isinstance(example_inputs[input_index], torch.Tensor):
                raise RuntimeError("[MegaKittens] Non-tensor inputs are not supported")
            out_tensors = (TensorMeta.from_torch(example_inputs[input_index]),)

        elif fx_node.op == "get_attr":
            is_input = True
            input_fx_nodes = ()  # in_nodes are never used for input nodes
            if len(fx_node.all_input_nodes) != 0:
                raise RuntimeError("[MegaKittens] Input node has source inputs.")
            try:
                attr = getattr(gm, fx_node.target)
            except Exception:
                raise RuntimeError("[MegaKittens] Invalid get_attr fx_node")
            if not isinstance(attr, torch.Tensor):
                raise RuntimeError("[MegaKittens] Non-tensor attributes are not supported")
            out_tensors = (TensorMeta.from_torch(attr),)

        elif fx_node.op in {"call_function", "call_module", "call_method"}:
            itype, aten_output_indices = convert_fx_node_to_itype(gm, fx_node)
            input_fx_nodes = fx_node.all_input_nodes
            out_tensors = extract_fx_node_outputs(fx_node, aten_output_indices=aten_output_indices)
            if aten_output_indices:
                output_idx_map[fx_node.name] = {aten_idx: itype_idx for itype_idx, aten_idx in enumerate(aten_output_indices)}

        elif fx_node.op == "output":
            is_output = True
            if not fx_node.args:
                raise RuntimeError("[MegaKittens] Output fx_node has no args")
            input_fx_nodes = flatten_fx_nodes(fx_node.args)  # AOTAutograd will convert flattened output list back into its original structure
            out_tensors = ()  # out_tensors are never used for output nodes

        else:
            raise RuntimeError(f"[MegaKittens] Invalid fx_node op {fx_node.op}")

        in_nodes: list[tuple[Node, int]] = []
        in_ranges: list[TensorRange] = []
        for input_fx_node in input_fx_nodes:
            if input_fx_node.op == "output":
                raise RuntimeError(f"[MegaKittens] Invalid input graph edge: fx_node '{fx_node.name}' consumes output fx_node '{input_fx_node.name}'")
            resolved_input_fx_node, output_idx, slice_chain = resolve_input_node(input_fx_node)
            if resolved_input_fx_node.name in output_idx_map:
                output_idx = output_idx_map[resolved_input_fx_node.name][output_idx]
            if resolved_input_fx_node.name not in node_by_name:  # because FX nodes are given in topological order
                raise RuntimeError(f"[MegaKittens] No source fx_node '{resolved_input_fx_node.name}' exists for '{fx_node.name}'")
            resolved_input_node = node_by_name[resolved_input_fx_node.name]
            if output_idx >= len(resolved_input_node.out_tensors):
                raise RuntimeError(
                    f"[MegaKittens] Node '{fx_node.name}' reads output slot {output_idx} of '{resolved_input_fx_node.name}',"
                    f" but source fx_node has {len(resolved_input_node.out_tensors)} outputs"
                )
            if is_output and slice_chain:  # potentially we *can* enable this; I just don't see a use case
                raise RuntimeError(
                    f"[MegaKittens] Output fx_node '{fx_node.name}' directly consumes a sliced view of"
                    f" '{resolved_input_fx_node.name}'; returning sliced views is not supported"
                )
            in_nodes.append((resolved_input_node, output_idx))
            in_ranges.append(TensorRange.from_slice_chain(resolved_input_node.out_tensors[output_idx].shape, slice_chain))

        node = Node(
            is_input=is_input,
            is_output=is_output,
            itype=itype,
            in_nodes=in_nodes,
            in_ranges=in_ranges,
            out_tensors=out_tensors,
            out_ranges=tuple(tensor_meta.full_range for tensor_meta in out_tensors),
            out_nodes=tuple([] for _ in out_tensors),  # empty lists, filled below
            input_index=input_index,
        )
        node_by_name[fx_node.name] = node
        dag_nodes.append(node)

        # Add this fx_node to each source fx_node's out_nodes.
        for input_node, output_idx in in_nodes:
            if output_idx >= len(input_node.out_nodes):
                raise RuntimeError(f"[MegaKittens] Output slot {output_idx} missing for source fx_node '{input_node}'")
            input_node.out_nodes[output_idx].append(node)

    if input_counter != len(example_inputs):
        raise RuntimeError(f"[MegaKittens] Number of input nodes is {input_counter}, but len(example_inputs) is {len(example_inputs)}")

    return DAG(dag_nodes)
