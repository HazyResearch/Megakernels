from __future__ import annotations

import operator
from typing import Any, Callable, Dict, List

import torch
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from torch.fx.passes.shape_prop import TensorMetadata

from . import utils
from .schema.dag import DAG, Node, OpType
from .schema.device import Device
from .schema.dtype import DType
from .schema.tensor import TensorMeta
from .dispatcher import Dispatcher
from .scheduler import schedule


def _resolve_optype(gm: torch.fx.GraphModule, node: torch.fx.Node) -> OpType:
    if node.op == "call_function":
        return OpType.from_call_function(node.target)
    if node.op == "call_method":
        return OpType.from_call_method(node.target)
    if node.op == "call_module":
        try:
            module = gm.get_submodule(node.target)
        except Exception:
            raise RuntimeError(f"[MegaKittens] Invalid call_module node '{node.name}' target={node.target!r}")
        return OpType.from_call_module(type(module))
    raise RuntimeError(f"[MegaKittens] Unsupported node op '{node.op}' for node '{node.name}'")


def _tensor_to_mk_tensor(node: torch.fx.Node, value: torch.Tensor) -> TensorMeta:
    if not value.is_contiguous():
        raise RuntimeError(
            f"[MegaKittens] Tensor for node '{node.name}' must be contiguous"
        )
    shape = tuple[int, ...](int(dim) for dim in value.shape)
    dtype = DType.from_torch(value.dtype)
    device = Device.from_torch(value.device)
    return TensorMeta(dtype=dtype, shape=shape, device=device)


def _is_contiguous(shape: tuple[int, ...], strides: tuple[int, ...]) -> bool:
    """Check if strides correspond to a contiguous (row-major) layout."""
    ndim = len(shape)
    if ndim == 0:
        return True
    expected = 1
    for i in range(ndim - 1, -1, -1):
        if shape[i] != 1 and strides[i] != expected:
            return False
        expected *= shape[i]
    return True


def _tensor_meta_to_mk_tensor(
    node: torch.fx.Node,
    tensor_meta: TensorMetadata,
    fallback_device: torch.device | None = None,
) -> TensorMeta:
    if tensor_meta.stride is not None and not _is_contiguous(
        tuple[int, ...](tensor_meta.shape), tuple(tensor_meta.stride)
    ):
        raise RuntimeError(
            f"[MegaKittens] Tensor for node '{node.name}' must be contiguous"
        )
    if tensor_meta.dtype is None:
        raise RuntimeError(f"[MegaKittens] Missing tensor metadata dtype for node '{node.name}'")
    shape = tuple[int, ...](int(dim) for dim in tensor_meta.shape)
    dtype = DType.from_torch(tensor_meta.dtype)
    tensor_meta_device = getattr(tensor_meta, "device", None)
    if tensor_meta_device is None:
        tensor_meta_device = fallback_device
    if tensor_meta_device is None:
        raise RuntimeError(f"[MegaKittens] Missing tensor metadata device for node '{node.name}'")
    device = Device.from_torch(tensor_meta_device)
    return TensorMeta(dtype=dtype, shape=shape, device=device)


def _get_output_tensors(node: torch.fx.Node) -> tuple[TensorMeta, ...]:
    if "tensor_meta" in node.meta:
        tensor_meta = node.meta["tensor_meta"]
        val = node.meta.get("val")
        if isinstance(tensor_meta, (tuple, list)):
            if all(isinstance(tm, TensorMetadata) for tm in tensor_meta):
                if val is None:
                    return tuple(_tensor_meta_to_mk_tensor(node, tm) for tm in tensor_meta)

                elif isinstance(val, torch.Tensor):
                    if len(tensor_meta) != 1:
                        raise RuntimeError(
                            f"[MegaKittens] Node '{node.name}' has {len(tensor_meta)} tensor metadata outputs"
                            f" but single Tensor 'val' provided"
                        )
                    return (_tensor_meta_to_mk_tensor(node, tensor_meta[0], fallback_device=val.device),)

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
                    _tensor_meta_to_mk_tensor(
                        node,
                        tm,
                        fallback_device=val[idx].device,
                    )
                    for idx, tm in enumerate(tensor_meta)
                )

            elif isinstance(val, torch.Tensor):
                return (_tensor_to_mk_tensor(node, val),)

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
            return (_tensor_meta_to_mk_tensor(node, tensor_meta, fallback_device=fallback_device),)
        raise RuntimeError(
            f"[MegaKittens] Unsupported tensor_meta type for node '{node.name}'"
            f" (type={type(tensor_meta).__name__})"
        )

    elif "val" in node.meta:
        value = node.meta["val"]
        if isinstance(value, (tuple, list)):
            tensors = []
            for v in value:
                if not isinstance(v, torch.Tensor):
                    raise RuntimeError(
                        f"[MegaKittens] Non-tensor entry in node metadata for node '{node.name}'"
                        f" (indexed entry type={type(v).__name__})"
                    )
                tensors.append(_tensor_to_mk_tensor(node, v))
            return tuple(tensors)
        elif isinstance(value, torch.Tensor):
            return (_tensor_to_mk_tensor(node, value),)
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


def fx_graph_to_mk_dag(
    gm: torch.fx.GraphModule,
    example_inputs: List[Any],
) -> DAG:
    """
    Convert an FX GraphModule plus example inputs into a MegaKittens node-centric DAG.

    Returns:
        A list of `dag.Node` values with topology represented by node connectivity fields.

    Dead nodes (not reachable from the output) are pruned.
    """
    all_graph_nodes = list(gm.graph.nodes)

    # Prune to only nodes that feed into the output.
    output_nodes = [n for n in all_graph_nodes if n.op == "output"]
    if len(output_nodes) != 1:
        raise RuntimeError(f"[MegaKittens] Number of output nodes is {len(output_nodes)}")
    output_node = output_nodes[0] # FX graph always produces 1 output node
    # TODO: support void functions
    if not output_node.args or output_node.args[0] is None:
        raise RuntimeError(
            "[MegaKittens] Void output graphs are not supported. "
            "Please return at least one tensor."
        )
    valid_names: set[str] = set()
    stack: List[torch.fx.Node] = [output_node]
    while stack:
        current = stack.pop()
        if current.name in valid_names:
            continue
        valid_names.add(current.name)
        stack.extend(current.all_input_nodes)
    graph_nodes = [
        n
        for n in all_graph_nodes
        if n.name in valid_names
        and not (n.op == "call_function" and n.target == operator.getitem)
    ]

    # Extract all DAG nodes
    _input_index: int = 0
    node_by_name: Dict[str, Node] = {}
    dag_nodes: List[Node] = []

    for node in graph_nodes:
        input_index: int | None = None
        out_tensors: tuple[TensorMeta, ...] | None = None

        if node.op == "placeholder":
            optype = OpType.input
            input_nodes = list(node.all_input_nodes)
            if _input_index >= len(example_inputs):
                raise RuntimeError("[MegaKittens] Number of input nodes is greater than len(example_inputs)")
            input_index = _input_index
            example_input = example_inputs[_input_index]
            _input_index += 1
            if isinstance(example_input, torch.Tensor):
                out_tensors = (_tensor_to_mk_tensor(node, example_input),)
            else:
                raise RuntimeError("[MegaKittens] Non-tensor inputs are not supported")

        elif node.op == "get_attr":
            optype = OpType.input
            input_nodes = list(node.all_input_nodes)
            try:
                attr = getattr(gm, node.target)
            except Exception:
                raise RuntimeError("[MegaKittens] Invalid get_attr node")
            if isinstance(attr, torch.Tensor):
                out_tensors = (_tensor_to_mk_tensor(node, attr),)
            else:
                raise RuntimeError("[MegaKittens] Non-tensor attributes are not supported")

        elif node.op in {"call_function", "call_module", "call_method"}:
            optype = _resolve_optype(gm, node)
            input_nodes = list(node.all_input_nodes)
            out_tensors = _get_output_tensors(node)

        elif node.op == "output":
            optype = OpType.output
            if not node.args:
                raise RuntimeError("[MegaKittens] Output node has no args")
            input_nodes = _flatten_output_nodes(node.args)
            if "tensor_meta" in node.meta or "val" in node.meta:
                out_tensors = _get_output_tensors(node)

        else:
            raise RuntimeError(f"[MegaKittens] Invalid node op {node.op}")

        in_nodes_list = []
        for input_node in input_nodes:
            if input_node.name not in valid_names:
                raise RuntimeError(
                    f"[MegaKittens] Invalid input '{input_node.name}' for node '{node.name}':"
                    " not reachable from graph output"
                )
            if input_node.op == "output":
                raise RuntimeError(
                    f"[MegaKittens] Invalid input graph edge: node '{node.name}' consumes output node '{input_node.name}'"
                )
            resolved_node, output_idx = _resolve_input_node_and_output_index(input_node)
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
            optype=optype,
            out_tensors=out_tensors,
            in_nodes=tuple(in_nodes_list),
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


def megakittens_backend(
    fn: Callable[..., Any],
    *,
    dry_run: bool = False,
    verify: bool = False,
    profile: bool = False,
    debug: bool = False,
    save_dag: bool = False,
    save_schedule: bool = False,
    use_jit_cache: bool = True,
) -> Callable[[torch.fx.GraphModule, List[Any]], Callable[..., Any]]:
    def _megakittens_backend(gm: torch.fx.GraphModule, example_inputs: List[Any]) -> Callable[..., Any]:
        if debug:
            print(f"[MegaKittens] Compiling function `{fn.__qualname__}`")
            print(f"[MegaKittens] FX graph:")
            gm.graph.print_tabular()

        if save_dag or save_schedule:
            base_path = utils.create_log_base_path(fn=fn)

        dag = fx_graph_to_mk_dag(gm, example_inputs)

        if save_dag:
            dag_json = utils.save_dag_as_png_as_json(dag, base_path)
            utils.save_dag_as_png(dag_json, base_path)

        if dry_run:
            if debug:
                print(f"[MegaKittens] Dry run mode; returning original function")
            return make_boxed_func(gm)

        (
            instruction_metas,
            tensor_metas,
            instructions,
            num_barriers,
            input_tensor_indices,
            output_tensor_indices,
        ) = schedule(dag)

        if save_schedule:
            utils.save_schedule_as_txt(
                tensor_metas, instructions, instruction_metas, num_barriers, base_path
            )

        dispatcher = Dispatcher(
            instruction_metas=instruction_metas,
            tensor_metas=tensor_metas,
            instructions=instructions,
            num_barriers=num_barriers,
            input_tensor_indices=input_tensor_indices,
            output_tensor_indices=output_tensor_indices,
            use_jit_cache=use_jit_cache,
        )

        return make_boxed_func(dispatcher)

    return aot_autograd(
        fw_compiler=_megakittens_backend,
        bw_compiler=_megakittens_backend,
    )
