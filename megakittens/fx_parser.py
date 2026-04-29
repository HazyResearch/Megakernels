from __future__ import annotations

import operator
from typing import Any, List

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2
from torch.fx.passes.shape_prop import TensorMetadata

from .schema.dag import DAG, Node
from .schema.itype import IType
from .schema.tensor import TensorMeta, TensorRange, TensorSlice


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


def resolve_and_build_call_node(
    fx_node: torch.fx.Node, ctx: dict,
    target: Any, args: Any, kwargs: Any, input_fx_nodes: list[torch.fx.Node],
) -> Node:
    result = IType.from_torch(target, args=args, kwargs=kwargs)
    if isinstance(result, tuple):
        itype, aten_output_indices = result
        if not isinstance(itype, IType) or not isinstance(aten_output_indices, list):
            raise RuntimeError(f"[MegaKittens] Invalid resolver result for node '{fx_node.name}': {result!r}")
    else:
        if not isinstance(result, IType):
            raise RuntimeError(f"[MegaKittens] Invalid resolve result for node '{fx_node.name}': expected IType, got {type(result).__name__}")
        itype, aten_output_indices = result, []

    out_tensors = extract_fx_node_outputs(fx_node, aten_output_indices=aten_output_indices)
    if aten_output_indices:
        ctx["aten_to_mk_output_idx"][fx_node.name] = {aten_idx: itype_idx for itype_idx, aten_idx in enumerate(aten_output_indices)}

    in_nodes: list[tuple[Node, int]] = []
    in_ranges: list[TensorRange] = []
    in_tensors: list[TensorMeta] = []
    for input_fx_node in input_fx_nodes:
        if input_fx_node.op == "output":
            raise RuntimeError(f"[MegaKittens] Invalid input graph edge: node '{fx_node.name}' consumes output node '{input_fx_node.name}'")
        if input_fx_node.name not in ctx["fx_name_to_node"]:
            raise RuntimeError(f"[MegaKittens] No source node '{input_fx_node.name}' for '{fx_node.name}'")
        dag_input = ctx["fx_name_to_node"][input_fx_node.name]
        if len(dag_input.out_tensors) != 1:
            raise RuntimeError(
                f"[MegaKittens] Node '{fx_node.name}' directly consumes multi-output node '{input_fx_node.name}'"
                f" (expected getitem in between)"
            )
        in_nodes.append((dag_input, 0))
        in_ranges.append(dag_input.out_tensors[0].full_range)
        in_tensors.append(dag_input.out_tensors[0])

    return Node(
        itype=itype,
        in_nodes=in_nodes,
        in_ranges=in_ranges,
        in_tensors=in_tensors,
        out_tensors=out_tensors,
        out_ranges=tuple(tm.full_range for tm in out_tensors),
        out_nodes=tuple([] for _ in out_tensors),
    )


def parse_placeholder_node(fx_node: torch.fx.Node, ctx: dict) -> Node:
    if len(fx_node.all_input_nodes) != 0:
        raise RuntimeError("[MegaKittens] Input node has source inputs.")
    if ctx["input_counter"] >= len(ctx["example_inputs"]):
        raise RuntimeError("[MegaKittens] Number of input nodes is greater than len(example_inputs)")
    if not isinstance(ctx["example_inputs"][ctx["input_counter"]], torch.Tensor):
        raise RuntimeError("[MegaKittens] Non-tensor inputs are not supported")
    out_tensors = (TensorMeta.from_torch(ctx["example_inputs"][ctx["input_counter"]]),)
    input_index = ctx["input_counter"]
    ctx["input_counter"] += 1
    return Node(
        is_input=True,
        in_nodes=(),
        in_ranges=(),
        in_tensors=(),
        out_tensors=out_tensors,
        out_ranges=tuple(tm.full_range for tm in out_tensors),
        out_nodes=tuple([] for _ in out_tensors),
        input_index=input_index,
    )


def parse_get_attr_node(fx_node: torch.fx.Node, ctx: dict) -> Node:
    if len(fx_node.all_input_nodes) != 0:
        raise RuntimeError("[MegaKittens] Input node has source inputs.")
    try:
        attr = getattr(ctx["gm"], fx_node.target)
    except Exception:
        raise RuntimeError("[MegaKittens] Invalid get_attr fx_node")
    if not isinstance(attr, torch.Tensor):
        raise RuntimeError("[MegaKittens] Non-tensor attributes are not supported")
    out_tensors = (TensorMeta.from_torch(attr),)
    return Node(
        is_input=True,
        in_nodes=(),
        in_ranges=(),
        in_tensors=(),
        out_tensors=out_tensors,
        out_ranges=tuple(tm.full_range for tm in out_tensors),
        out_nodes=tuple([] for _ in out_tensors),
    )


def parse_getitem_node(fx_node: torch.fx.Node, ctx: dict) -> Node:
    args = fx_node.args
    if len(args) < 2:
        raise RuntimeError(f"[MegaKittens] getitem expects at least 2 args for node '{fx_node.name}', got {len(args)}")
    source_fx_node = args[0]
    aten_idx = args[1]
    if not isinstance(source_fx_node, torch.fx.Node):
        raise RuntimeError(f"[MegaKittens] getitem source is not an FX node for '{fx_node.name}'")
    if not isinstance(aten_idx, int):
        raise RuntimeError(f"[MegaKittens] getitem index is not int for '{fx_node.name}'")
    if source_fx_node.name not in ctx["fx_name_to_node"]:
        raise RuntimeError(f"[MegaKittens] No source node '{source_fx_node.name}' for getitem '{fx_node.name}'")
    source_node = ctx["fx_name_to_node"][source_fx_node.name]
    output_idx = aten_idx
    if source_fx_node.name in ctx["aten_to_mk_output_idx"]:
        output_idx = ctx["aten_to_mk_output_idx"][source_fx_node.name][aten_idx]
    if output_idx >= len(source_node.out_tensors):
        raise RuntimeError(
            f"[MegaKittens] getitem index {output_idx} out of range for '{source_fx_node.name}'"
            f" with {len(source_node.out_tensors)} outputs"
        )
    out_tensors = (source_node.out_tensors[output_idx],)
    return Node(
        itype="getitem",
        in_nodes=((source_node, output_idx),),
        in_ranges=(source_node.out_tensors[output_idx].full_range,),
        in_tensors=(source_node.out_tensors[output_idx],),
        out_tensors=out_tensors,
        out_ranges=tuple(tm.full_range for tm in out_tensors),
        out_nodes=tuple([] for _ in out_tensors),
    )


def parse_select_node(fx_node: torch.fx.Node, ctx: dict) -> Node:
    args = fx_node.args
    if len(args) != 3:
        raise RuntimeError(f"[MegaKittens] aten.select.int expects 3 args for node '{fx_node.name}', got {len(args)}")
    if not isinstance(args[0], torch.fx.Node):
        raise RuntimeError(f"[MegaKittens] aten.select.int args[0] is not an FX node for '{fx_node.name}'")
    if not isinstance(args[1], int) or not isinstance(args[2], int):
        raise RuntimeError(
            f"[MegaKittens] aten.select.int requires int dim/idx for node '{fx_node.name}',"
            f" got dim={args[1]!r} idx={args[2]!r}"
        )
    source_fx_node, dim, idx = args[0], args[1], args[2]
    if source_fx_node.name not in ctx["fx_name_to_node"]:
        raise RuntimeError(f"[MegaKittens] No source node '{source_fx_node.name}' for select '{fx_node.name}'")
    source_node = ctx["fx_name_to_node"][source_fx_node.name]
    in_range = TensorRange.from_slice_chain(source_node.out_tensors[0].shape, [TensorSlice(op="select", dim=dim, start=idx, end=idx + 1)])
    out_tensors = extract_fx_node_outputs(fx_node)
    return Node(
        itype="select",
        in_nodes=((source_node, 0),),
        in_ranges=(in_range,),
        in_tensors=(source_node.out_tensors[0],),
        out_tensors=out_tensors,
        out_ranges=tuple(tm.full_range for tm in out_tensors),
        out_nodes=tuple([] for _ in out_tensors),
    )


def parse_slice_node(fx_node: torch.fx.Node, ctx: dict) -> Node:
    args = fx_node.args
    if not (2 <= len(args) <= 5):
        raise RuntimeError(f"[MegaKittens] aten.slice.Tensor expects 2-5 args for node '{fx_node.name}', got {len(args)}")
    if not isinstance(args[0], torch.fx.Node):
        raise RuntimeError(f"[MegaKittens] aten.slice.Tensor parent is not an FX node for '{fx_node.name}'")
    dim = args[1]
    start = args[2] if len(args) > 2 else None
    end = args[3] if len(args) > 3 else None
    step = args[4] if len(args) > 4 and args[4] is not None else 1
    if not isinstance(dim, int):
        raise RuntimeError(f"[MegaKittens] aten.slice.Tensor requires int dim for node '{fx_node.name}', got dim={dim!r}")
    if start is not None and not isinstance(start, int):
        raise RuntimeError(f"[MegaKittens] aten.slice.Tensor start must be int or None for node '{fx_node.name}', got start={start!r}")
    if end is not None and not isinstance(end, int):
        raise RuntimeError(f"[MegaKittens] aten.slice.Tensor end must be int or None for node '{fx_node.name}', got end={end!r}")
    if step != 1:
        raise RuntimeError(f"[MegaKittens] aten.slice.Tensor step != 1 is not supported for node '{fx_node.name}'")
    source_fx_node = args[0]
    if source_fx_node.name not in ctx["fx_name_to_node"]:
        raise RuntimeError(f"[MegaKittens] No source node '{source_fx_node.name}' for slice '{fx_node.name}'")
    source_node = ctx["fx_name_to_node"][source_fx_node.name]
    in_range = TensorRange.from_slice_chain(source_node.out_tensors[0].shape, [TensorSlice(op="slice", dim=dim, start=start, end=end)])
    out_tensors = extract_fx_node_outputs(fx_node)
    return Node(
        itype="slice",
        in_nodes=((source_node, 0),),
        in_ranges=(in_range,),
        in_tensors=(source_node.out_tensors[0],),
        out_tensors=out_tensors,
        out_ranges=tuple(tm.full_range for tm in out_tensors),
        out_nodes=tuple([] for _ in out_tensors),
    )


def parse_auto_functionalized_node(fx_node: torch.fx.Node, ctx: dict) -> Node:
    underlying_fx_node = fx_node.args[0]
    canonical_args = underlying_fx_node._schema.arguments
    mutated_args = list(fx_node.kwargs.get("_all_bases", []))
    reconstructed_args: list = []
    input_fx_nodes: list[torch.fx.Node] = []
    for canonical_arg in canonical_args:
        base_index_key = f"_{canonical_arg.name}_base_index"
        length_key = f"_{canonical_arg.name}_length"
        if base_index_key in fx_node.kwargs:
            input_node = mutated_args[fx_node.kwargs[base_index_key]]
            reconstructed_args.append(input_node)
            input_fx_nodes.append(input_node)
        elif length_key in fx_node.kwargs:
            length = fx_node.kwargs[length_key]
            input_node_list: list[torch.fx.Node] = []
            for i in range(length):
                input_node = mutated_args[fx_node.kwargs[f"_{canonical_arg.name}_{i}_base_index"]]
                input_fx_nodes.append(input_node)
                input_node_list.append(input_node)
            reconstructed_args.append(input_node_list)
        elif canonical_arg.name in fx_node.kwargs:
            val = fx_node.kwargs[canonical_arg.name]
            reconstructed_args.append(val)
            if isinstance(val, torch.fx.Node):
                input_fx_nodes.append(val)
            elif isinstance(val, (list, tuple)):
                for v in val:
                    if not isinstance(v, torch.fx.Node):
                        raise RuntimeError("[MegaKittens] Invalid auto_functionalized_v2 argument")
                    input_fx_nodes.append(v)
            else:
                raise RuntimeError("[MegaKittens] Invalid auto_functionalized_v2 argument")
        else:
            raise RuntimeError("[MegaKittens] Invalid auto_functionalized_v2 argument")
    return resolve_and_build_call_node(fx_node, ctx, underlying_fx_node, reconstructed_args, {}, input_fx_nodes)


def parse_call_function_node(fx_node: torch.fx.Node, ctx: dict) -> Node:
    return resolve_and_build_call_node(fx_node, ctx, fx_node.target, fx_node.args, fx_node.kwargs, list(fx_node.all_input_nodes))


def parse_call_module_node(fx_node: torch.fx.Node, ctx: dict) -> Node:
    try:
        target = type(ctx["gm"].get_submodule(fx_node.target))
    except Exception:
        raise RuntimeError(f"[MegaKittens] Invalid call_module node '{fx_node.name}' target={fx_node.target!r}")
    return resolve_and_build_call_node(fx_node, ctx, target, fx_node.args, fx_node.kwargs, list(fx_node.all_input_nodes))


def parse_output_node(fx_node: torch.fx.Node, ctx: dict) -> Node:
    if not fx_node.args:
        raise RuntimeError("[MegaKittens] Output fx_node has no args")
    in_nodes: list[tuple[Node, int]] = []
    in_ranges: list[TensorRange] = []
    in_tensors: list[TensorMeta] = []
    stack = list(fx_node.args)
    while stack:
        val = stack.pop()
        if isinstance(val, torch.fx.Node):
            if val.op == "output":
                raise RuntimeError(f"[MegaKittens] Invalid input graph edge: output '{fx_node.name}' consumes output '{val.name}'")
            if val.name not in ctx["fx_name_to_node"]:
                raise RuntimeError(f"[MegaKittens] No source node '{val.name}' for output '{fx_node.name}'")
            dag_input = ctx["fx_name_to_node"][val.name]
            if len(dag_input.out_tensors) != 1:
                raise RuntimeError(
                    f"[MegaKittens] Output node '{fx_node.name}' directly consumes multi-output node '{val.name}'"
                    f" (expected getitem in between)"
                )
            in_nodes.append((dag_input, 0))
            in_ranges.append(dag_input.out_tensors[0].full_range)
            in_tensors.append(dag_input.out_tensors[0])
        elif isinstance(val, (list, tuple)):
            stack.extend(val)
        elif isinstance(val, dict):
            stack.extend(val.values())
        else:
            raise RuntimeError(f"[MegaKittens] Unsupported output arg type '{type(val).__name__}'")
    in_nodes.reverse()
    in_ranges.reverse()
    in_tensors.reverse()
    return Node(
        is_output=True,
        in_nodes=in_nodes,
        in_ranges=in_ranges,
        in_tensors=in_tensors,
        out_tensors=(),
        out_ranges=(),
        out_nodes=()
    )


def parse_transpose_node(fx_node: torch.fx.Node, ctx: dict) -> Node:
    args = fx_node.args
    if len(args) < 1:
        raise RuntimeError(f"[MegaKittens] transpose op expects at least 1 arg for node '{fx_node.name}', got {len(args)}")
    if not isinstance(args[0], torch.fx.Node):
        raise RuntimeError(f"[MegaKittens] transpose op source is not an FX node for '{fx_node.name}'")
    source_fx_node = args[0]
    if source_fx_node.name not in ctx["fx_name_to_node"]:
        raise RuntimeError(f"[MegaKittens] No source node '{source_fx_node.name}' for transpose op '{fx_node.name}'")
    source_node = ctx["fx_name_to_node"][source_fx_node.name]
    if len(source_node.out_tensors) != 1:
        raise RuntimeError(
            f"[MegaKittens] transpose op '{fx_node.name}' directly consumes multi-output node '{source_fx_node.name}'"
            f" (expected getitem in between)"
        )
    src_meta = source_node.out_tensors[0]
    ndim = len(src_meta.shape)
    pad = TensorRange.NUM_DIMS - ndim
    if fx_node.target == torch.ops.aten.t.default:
        if ndim != 2:
            raise RuntimeError(f"[MegaKittens] aten.t requires 2D tensor, got {ndim}D")
        permutation = (1, 0)
    elif fx_node.target == torch.ops.aten.permute.default:
        permutation = tuple(args[1])
    else:
        raise RuntimeError(f"[MegaKittens] Unknown transpose op: {fx_node.target}")
    transposed_shape = tuple(src_meta.shape[p] for p in permutation)
    perm_4d = tuple(range(pad)) + tuple(pad + p for p in permutation)
    out_tensors = (TensorMeta(dtype=src_meta.dtype, shape=transposed_shape, device=src_meta.device),)
    return Node(
        itype=f"transpose:{','.join(str(p) for p in perm_4d)}",
        in_nodes=((source_node, 0),),
        in_ranges=(source_node.out_tensors[0].full_range,),
        in_tensors=(source_node.out_tensors[0],),
        out_tensors=out_tensors,
        out_ranges=tuple(tm.full_range for tm in out_tensors),
        out_nodes=tuple([] for _ in out_tensors),
    )


def parse_reduce_node(fx_node: torch.fx.Node, ctx: dict) -> list[Node]:
    """Generic handler for reduction ops. When keepdim=True, decomposes into reduce + view."""
    args, kwargs = fx_node.args, fx_node.kwargs
    keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)

    reduce_node = resolve_and_build_call_node(
        fx_node, ctx, fx_node.target, args, kwargs, list(fx_node.all_input_nodes),
    )

    if not keepdim:
        return [reduce_node]

    else:
        keepdim_tms = extract_fx_node_outputs(fx_node)
        if len(keepdim_tms) != 1:
            raise RuntimeError(
                f"[MegaKittens] Reduce with keepdim expects 1 output, got {len(keepdim_tms)} for node '{fx_node.name}'"
            )
        dims = args[1] if len(args) > 1 else kwargs.get("dim", [-1])
        ndim = len(keepdim_tms[0].shape)
        reduced_dims = {d % ndim for d in (dims if isinstance(dims, (list, tuple)) else [dims])}  # convert negative indices
        no_keepdim_shape = tuple(s for i, s in enumerate(keepdim_tms[0].shape) if i not in reduced_dims)
        reduce_meta = TensorMeta(dtype=keepdim_tms[0].dtype, shape=no_keepdim_shape, device=keepdim_tms[0].device)

        # Modify reduce node to have no keepdim output
        reduce_node.out_tensors = (reduce_meta,)
        reduce_node.out_ranges = (reduce_meta.full_range,)
        reduce_node.out_nodes = ([],)

        view_node = Node(
            itype="view",
            in_nodes=((reduce_node, 0),),
            in_ranges=(reduce_meta.full_range,),
            in_tensors=(reduce_meta,),
            out_tensors=keepdim_tms,
            out_ranges=tuple(tm.full_range for tm in keepdim_tms),
            out_nodes=tuple([] for _ in keepdim_tms),
        )

        return [reduce_node, view_node]


def parse_view_node(fx_node: torch.fx.Node, ctx: dict) -> Node:
    args = fx_node.args
    if len(args) < 2:
        raise RuntimeError(f"[MegaKittens] view op expects at least 2 args for node '{fx_node.name}', got {len(args)}")
    if not isinstance(args[0], torch.fx.Node):
        raise RuntimeError(f"[MegaKittens] view op source is not an FX node for '{fx_node.name}'")
    source_fx_node = args[0]
    if source_fx_node.name not in ctx["fx_name_to_node"]:
        raise RuntimeError(f"[MegaKittens] No source node '{source_fx_node.name}' for view op '{fx_node.name}'")
    source_node = ctx["fx_name_to_node"][source_fx_node.name]
    if len(source_node.out_tensors) != 1:
        raise RuntimeError(
            f"[MegaKittens] view op '{fx_node.name}' directly consumes multi-output node '{source_fx_node.name}'"
            f" (expected getitem in between)"
        )
    out_tensors = extract_fx_node_outputs(fx_node)
    return Node(
        itype="view",
        in_nodes=((source_node, 0),),
        in_ranges=(source_node.out_tensors[0].full_range,),
        in_tensors=(source_node.out_tensors[0],),
        out_tensors=out_tensors,
        out_ranges=tuple(tm.full_range for tm in out_tensors),
        out_nodes=tuple([] for _ in out_tensors),
    )


def extract_dag_from_fx_graph(gm: torch.fx.GraphModule, example_inputs: List[Any]) -> DAG:
    """Convert an FX GraphModule plus example inputs into an unoptimized MegaKittens DAG."""
    fx_nodes = list(gm.graph.nodes)
    output_nodes = [node for node in fx_nodes if node.op == "output"]
    if len(output_nodes) != 1:
        raise RuntimeError(f"[MegaKittens] Number of output nodes is {len(output_nodes)}")
    output_node = output_nodes[0]
    if not output_node.args or len(output_node.args) != 1 or output_node.args[0] is None:  # void functions not supported by aot itself
        raise RuntimeError("[MegaKittens] Void output graphs are not supported. Please return at least one tensor.")

    # FX graph parsing context
    ctx: dict = {
        "gm": gm,
        "example_inputs": example_inputs,
        "input_counter": 0,
        "aten_to_mk_output_idx": {},
        "fx_name_to_node": {},
    }
    dag_nodes: List[Node] = []

    # Main parsing loop
    for fx_node in fx_nodes:
        if fx_node.op == "placeholder":
            node = parse_placeholder_node(fx_node, ctx)
        elif fx_node.op == "get_attr":
            node = parse_get_attr_node(fx_node, ctx)
        elif fx_node.op == "call_function" and fx_node.target == operator.getitem:
            node = parse_getitem_node(fx_node, ctx)
        elif fx_node.op == "call_function" and fx_node.target == torch.ops.aten.select.int:
            node = parse_select_node(fx_node, ctx)
        elif fx_node.op == "call_function" and fx_node.target == torch.ops.aten.slice.Tensor:
            node = parse_slice_node(fx_node, ctx)
        elif fx_node.op == "call_function" and fx_node.target in {
            torch.ops.aten.t.default, torch.ops.aten.permute.default,
        }:
            node = parse_transpose_node(fx_node, ctx)
        elif fx_node.op == "call_function" and fx_node.target in {
            torch.ops.aten.view.default, torch.ops.aten.reshape.default, torch.ops.aten._unsafe_view.default,
        }:
            node = parse_view_node(fx_node, ctx)
        elif fx_node.op == "call_function" and fx_node.target in {
            torch.ops.aten.mean.dim, torch.ops.aten.sum.dim_IntList, torch.ops.aten.amax.default, torch.ops.aten.amin.default,
        }:
            nodes = parse_reduce_node(fx_node, ctx)
            for n in nodes[:-1]:
                dag_nodes.append(n)
            node = nodes[-1]  # to make `dag_nodes.append(node)` below work
        elif fx_node.op == "call_function" and fx_node.target is auto_functionalized_v2:
            node = parse_auto_functionalized_node(fx_node, ctx)
        elif fx_node.op in {"call_function", "call_method"}:
            node = parse_call_function_node(fx_node, ctx)
        elif fx_node.op == "call_module":
            node = parse_call_module_node(fx_node, ctx)
        elif fx_node.op == "output":
            node = parse_output_node(fx_node, ctx)
        else:
            raise RuntimeError(f"[MegaKittens] Invalid fx_node op {fx_node.op}")

        ctx["fx_name_to_node"][fx_node.name] = node
        dag_nodes.append(node)

    # Append output node info
    for node in dag_nodes:
        for in_node, output_idx in node.in_nodes:
            if output_idx >= len(in_node.out_nodes):
                raise RuntimeError(f"[MegaKittens] Output slot {output_idx} missing for source node '{in_node}'")
            in_node.out_nodes[output_idx].append(node)

    if ctx["input_counter"] != len(example_inputs):
        raise RuntimeError(f"[MegaKittens] Number of input nodes is {ctx['input_counter']}, but len(example_inputs) is {len(example_inputs)}")

    return DAG(dag_nodes)
