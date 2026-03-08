from __future__ import annotations

from enum import Enum
from typing import List, Literal, Tuple

from pydantic import BaseModel, Field, NonNegativeInt


class DType(str, Enum):
    int64 = "int64"
    int32 = "int32"
    int16 = "int16"
    int8 = "int8"
    fp64 = "fp64"
    fp32 = "fp32"
    bf16 = "bf16"
    half = "half"
    fp8e4m3 = "fp8e4m3"
    fp8e5m2 = "fp8e5m2"
    fp8e8m0 = "fp8e8m0"
    fp4e2m1x2 = "fp4e2m1x2"


class OpType(str, Enum):
    input = "input"
    add = "add"
    matmul = "matmul"
    relu = "relu"
    output = "output"


class Device(BaseModel):
    type: Literal["cpu", "cuda"]
    index: int | None = Field(ge=0, le=7, default=None)

    def __str__(self) -> str:
        return f"{self.type}:{self.index}"

    model_config = {"frozen": True}


class TensorMeta(BaseModel):
    dtype: DType
    shape: Tuple[NonNegativeInt, ...] # TODO: support dynamic shapes
    device: Device


class Node(BaseModel):
    """
    Graph vertex for the DAG. This schema is node-centric (no separate Edge objects).
    """
    optype: OpType
    in_nodes: Tuple[Tuple[Node, NonNegativeInt], ...]
    out_tensors: Tuple[TensorMeta, ...]
    out_nodes: Tuple[List[Node], ...]

    # Op-specific fields
    input_index: int | None # None if not an input
    # TODO: support default values


def _validate_topological_dag(dag_nodes: List[Node]) -> None:
    node_index: dict[int, int] = {id(node): idx for idx, node in enumerate(dag_nodes)}
    for node_idx, node in enumerate(dag_nodes):
        for in_node, input_idx in node.in_nodes:
            parent_index = node_index.get(id(in_node))
            if parent_index is None:
                raise RuntimeError(
                    f"[MegaKittens] Invalid DAG connectivity: node at index {node_idx} has missing parent"
                )
            if parent_index >= node_idx:
                raise RuntimeError(
                    f"[MegaKittens] Invalid DAG topology at index {node_idx}: parent node appears after child"
                )
            if input_idx >= len(in_node.out_nodes):
                raise RuntimeError(
                    f"[MegaKittens] Invalid DAG connectivity: node index {node_idx} uses invalid source output slot {input_idx}"
                )
            if node not in in_node.out_nodes[input_idx]:
                raise RuntimeError(
                    f"[MegaKittens] Invalid DAG connectivity: edge from node index {parent_index}"
                    f" output slot {input_idx} to node index {node_idx} is missing"
                )

        for out_idx, out_nodes in enumerate(node.out_nodes):
            for out_node in out_nodes:
                out_node_idx = node_index.get(id(out_node))
                if out_node_idx is None:
                    raise RuntimeError(
                        f"[MegaKittens] Invalid DAG connectivity: edge from node index {node_idx}"
                        f" to unknown node (output slot {out_idx})"
                    )
                if not any(
                    id(in_node) == id(node) and input_idx == out_idx
                    for in_node, input_idx in out_node.in_nodes
                ):
                    raise RuntimeError(
                        f"[MegaKittens] Invalid DAG connectivity: node index {node_idx}"
                        f" is not registered as input to node index {out_node_idx}"
                    )


def validate_dag(dag_nodes: List[Node]) -> None:
    if not isinstance(dag_nodes, list):
        raise RuntimeError("[MegaKittens] DAG payload is not a list")

    for node in dag_nodes:
        if not isinstance(node, Node):
            raise RuntimeError("[MegaKittens] DAG payload contains non-Node entry")

    input_nodes = [node for node in dag_nodes if node.optype == OpType.input]
    output_nodes = [node for node in dag_nodes if node.optype == OpType.output]

    for node in input_nodes:
        if len(node.in_nodes) != 0:
            raise RuntimeError("[MegaKittens] Input node has inbound edges")

    for node in output_nodes:
        if len(node.out_nodes) != 0:
            raise RuntimeError("[MegaKittens] Output node has outbound edges")

    if len(output_nodes) != 1:
        raise RuntimeError(f"[MegaKittens] Number of output nodes is {len(output_nodes)}")

    _validate_topological_dag(dag_nodes)
