from __future__ import annotations

import operator
from typing import Any, Callable, Dict, List, Tuple

import torch
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from torch.fx.passes.shape_prop import TensorMetadata

from .dag import DType, Device, Edge, Node, OpType


_DTYPE_MAP = {
    torch.float64: DType.fp64,
    torch.float32: DType.fp32,
    torch.bfloat16: DType.bf16,
    torch.float16: DType.half,
    torch.float8_e4m3fn: DType.fp8e4m3,
    torch.float8_e5m2fnuz: DType.fp8e5m2,
    torch.float8_e8m0fnu: DType.fp8e8m0,
    torch.float4_e2m1fn_x2: DType.fp4e2m1x2,
}

_CALL_FUNCTION_OPS = {
    torch.add: OpType.add,
    torch.matmul: OpType.matmul,
    torch.mm: OpType.matmul,
    torch.relu: OpType.relu,
    operator.add: OpType.add,
    operator.matmul: OpType.matmul,
    torch.ops.aten.add: OpType.add,
    torch.ops.aten.add.default: OpType.add,
    torch.ops.aten.matmul: OpType.matmul,
    torch.ops.aten.matmul.default: OpType.matmul,
    torch.ops.aten.relu: OpType.relu,
    torch.ops.aten.relu.default: OpType.relu,
}

_CALL_METHOD_OPS = {
    "add": OpType.add,
    "matmul": OpType.matmul,
    "relu": OpType.relu,
}

_CALL_MODULE_OPS = {
    torch.nn.ReLU: OpType.relu,
    torch.nn.ReLU6: OpType.relu,
}


def _torch_dtype_to_mk_dtype(node: torch.fx.Node, dtype: torch.dtype) -> DType:
    mapped_dtype = _DTYPE_MAP.get(dtype)
    if mapped_dtype is None:
        raise RuntimeError(f"[MegaKittens] Unsupported dtype {dtype} in node '{node.name}'")
    return mapped_dtype


def _torch_device_to_mk_device(value: torch.device) -> Device:
    index = value.index if value.index is not None else torch.cuda.current_device()
    return Device(type=value.type, index=index)


def _resolve_optype(gm: torch.fx.GraphModule, node: torch.fx.Node) -> OpType:
    if node.op == "call_function":
        if node.target in _CALL_FUNCTION_OPS:
            return _CALL_FUNCTION_OPS[node.target]
        raise RuntimeError(
            f"[MegaKittens] Unsupported function op node '{node.name}' target={node.target!r}"
        )

    if node.op == "call_method":
        if node.target in _CALL_METHOD_OPS:
            return _CALL_METHOD_OPS[node.target]
        raise RuntimeError(
            f"[MegaKittens] Unsupported method op node '{node.name}' target={node.target!r}"
        )

    if node.op == "call_module":
        try:
            module = gm.get_submodule(node.target)
        except Exception:
            raise RuntimeError(f"[MegaKittens] Invalid call_module node '{node.name}' target={node.target!r}")
        module_type = type(module)
        if module_type in _CALL_MODULE_OPS:
            return _CALL_MODULE_OPS[module_type]
        raise RuntimeError(
            f"[MegaKittens] Unsupported module op node '{node.name}' module={type(module).__name__}"
        )

    raise RuntimeError(f"[MegaKittens] Unsupported node op '{node.op}' for node '{node.name}'")


def fx_graph_to_mk_dag(
    gm: torch.fx.GraphModule,
    example_inputs: List[Any],
) -> Tuple[List[Node], List[Edge]]:
    """
    Convert an FX GraphModule plus example inputs into a MegaKittens DAG.

    Returns (nodes, edges) where:
        - nodes are dag.Node values (placeholders, get_attr, intermediates)
        - edges are dag.Edge data dependencies labeled by dag.OpType

    Dead nodes (not reachable from the output) are pruned.
    """
    all_graph_nodes = list(gm.graph.nodes)

    # Prune to only nodes that feed into the output.
    output_nodes = [n for n in all_graph_nodes if n.op == "output"]
    if len(output_nodes) != 1:
        raise RuntimeError(f"[MegaKittens] Number of output nodes is {len(output_nodes)}")
    output_node = output_nodes[0] # TODO: support void functions
    valid_names: set[str] = set()
    stack: list[torch.fx.Node] = [output_node]
    while stack:
        current = stack.pop()
        if current.name in valid_names:
            continue
        valid_names.add(current.name)
        stack.extend(current.all_input_nodes)
    graph_nodes = [n for n in all_graph_nodes if n.name in valid_names and n.op != "output"]

    # Extract DAG nodes (values)
    _input_index: int = 0
    dag_nodes: list[Node] = []
    node_by_name: Dict[str, Node] = {}

    for node in graph_nodes:
        input_index: int = -1
        dtype: DType | None = None
        shape: tuple[int, ...] | None = None
        device: Device | None = None

        if node.op == "placeholder":
            if _input_index >= len(example_inputs):
                raise RuntimeError("[MegaKittens] Number of input nodes is greater than len(example_inputs)")
            input_index = _input_index
            example_input = example_inputs[_input_index]
            _input_index += 1
            if isinstance(example_input, torch.Tensor):
                shape = tuple[int, ...](int(dim) for dim in example_input.shape)
                dtype = _torch_dtype_to_mk_dtype(node, example_input.dtype)
                device = _torch_device_to_mk_device(example_input.device)
            else:
                raise RuntimeError(f"[MegaKittens] Non-tensor inputs are not supported")

        elif node.op == "get_attr":
            try:
                attr = getattr(gm, node.target)
            except Exception:
                raise RuntimeError("[MegaKittens] Invalid get_attr node")
            if isinstance(attr, torch.Tensor):
                shape = tuple[int, ...](int(dim) for dim in attr.shape)
                dtype = _torch_dtype_to_mk_dtype(node, attr.dtype)
                device = _torch_device_to_mk_device(attr.device)
            else:
                raise RuntimeError(f"[MegaKittens] Non-tensor attributes are not supported")

        elif node.op in {"call_function", "call_module", "call_method"}:
            if "tensor_meta" in node.meta:
                tensor_meta: TensorMetadata = node.meta["tensor_meta"]
                shape = tuple[int, ...](int(dim) for dim in tensor_meta.shape)
                dtype = _torch_dtype_to_mk_dtype(node, tensor_meta.dtype)
                tensor_meta_device = getattr(tensor_meta, "device", None)
                if tensor_meta_device is None:
                    raise RuntimeError(
                        f"[MegaKittens] Missing tensor metadata device for node '{node.name}'"
                        f" (op={node.op}, target={node.target!r}, meta_keys={list(node.meta.keys())})"
                    )
                device = _torch_device_to_mk_device(tensor_meta_device)

            elif "val" in node.meta:
                if not isinstance(node.meta["val"], torch.Tensor):
                    raise RuntimeError("[MegaKittens] Node metadata is not a Torch Tensor")
                val = node.meta["val"]
                shape = tuple(int(dim) for dim in val.shape)
                dtype = _torch_dtype_to_mk_dtype(node, val.dtype)
                device = _torch_device_to_mk_device(val.device)

            else:
                raise RuntimeError(
                    f"[MegaKittens] Missing tensor metadata for node '{node.name}'"
                    f" (op={node.op}, target={node.target!r}, meta_keys={list(node.meta.keys())})"
                )

        else:
            raise RuntimeError(f"[MegaKittens] Invalid node op {node.op}")

        dag_node = Node(
            input_index=input_index,
            dtype=dtype,
            shape=shape,
            device=device,
        )
        dag_nodes.append(dag_node)
        node_by_name[node.name] = dag_node

    if _input_index != len(example_inputs):
        raise RuntimeError(
            f"[MegaKittens] Number of input nodes is {_input_index}, but len(example_inputs) is {len(example_inputs)}"
        )

    # Extract DAG edges (operations)
    dag_edges: list[Edge] = []

    for node in graph_nodes:
        if node.op in {"placeholder", "get_attr"}:
            continue

        optype = _resolve_optype(gm, node)

        if node.name not in node_by_name:
            raise RuntimeError("[MegaKittens] No destination node exists")
        dst = (node_by_name.get(node.name),)

        src = []
        for input_node in node.all_input_nodes:
            if input_node.name not in node_by_name:
                raise RuntimeError("[MegaKittens] No source node exists")
            src.append(node_by_name.get(input_node.name))
        src = tuple(src)

        dag_edges.append(Edge(optype=optype, in_nodes=src, out_nodes=dst))

    return (dag_nodes, dag_edges)


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
            nodes, edges = fx_graph_to_mk_dag(gm, example_inputs)
            _save_dag(nodes, edges, fn=fn)

        return make_boxed_func(gm.forward)

    return aot_autograd(
        fw_compiler=_megakittens_backend,
        bw_compiler=_megakittens_backend,
    )
