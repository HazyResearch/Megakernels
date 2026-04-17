from __future__ import annotations

import builtins
from typing import List, Tuple

from pydantic import BaseModel, NonNegativeInt

from .itype import IType
from .tensor import TensorMeta, TensorRange


class Node(BaseModel):
    """
    Graph vertex for the DAG. This schema is node-centric (no separate Edge objects).
    """
    model_config = {"arbitrary_types_allowed": True}
    is_input: bool = False
    is_output: bool = False  # There should be only 1 output node

    itype: IType | None = None  # None if input/output

    in_nodes: Tuple[Tuple[Node, NonNegativeInt], ...]  # [[source_node, output_slot_idx], ...]
    in_ranges: Tuple[TensorRange, ...]
    out_tensors: Tuple[TensorMeta, ...]
    out_ranges: Tuple[TensorRange, ...]
    out_nodes: Tuple[List[Node], ...]  # [[destination_node, ...], ...]

    # Op-specific fields
    # TODO: support default values
    input_index: int | None = None  # None if not an input

    # Unique identifier for this node
    id: int = 0
    def model_post_init(self, _) -> None:
        self.id = builtins.id(self)


class DAG:
    """Directed acyclic graph of compute nodes."""

    def __init__(self, nodes: List[Node]) -> None:
        self.nodes = nodes
        self.validate()

    def _validate_topological(self) -> None:
        node_index: dict[int, int] = {node.id: idx for idx, node in enumerate(self.nodes)}
        for node_idx, node in enumerate(self.nodes):
            for in_node, input_idx in node.in_nodes:
                parent_index = node_index.get(in_node.id)
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
                if not any(n.id == node.id for n in in_node.out_nodes[input_idx]):
                    raise RuntimeError(
                        f"[MegaKittens] Invalid DAG connectivity: edge from node index {parent_index}"
                        f" output slot {input_idx} to node index {node_idx} is missing"
                    )

            for out_idx, out_nodes in enumerate(node.out_nodes):
                for out_node in out_nodes:
                    out_node_idx = node_index.get(out_node.id)
                    if out_node_idx is None:
                        raise RuntimeError(
                            f"[MegaKittens] Invalid DAG connectivity: edge from node index {node_idx}"
                            f" to unknown node (output slot {out_idx})"
                        )
                    if not any(
                        in_node.id == node.id and input_idx == out_idx
                        for in_node, input_idx in out_node.in_nodes
                    ):
                        raise RuntimeError(
                            f"[MegaKittens] Invalid DAG connectivity: node index {node_idx}"
                            f" is not registered as input to node index {out_node_idx}"
                        )

    def validate(self) -> None:
        if not isinstance(self.nodes, list):
            raise RuntimeError("[MegaKittens] DAG payload is not a list")

        for node in self.nodes:
            if not isinstance(node, Node):
                raise RuntimeError("[MegaKittens] DAG payload contains non-Node entry")
            if node.is_input + node.is_output + (node.itype is not None) != 1:  # XOR
                raise RuntimeError("[MegaKittens] Node must be exactly one of: input, output, or itype")
            if len(node.out_nodes) != len(node.out_tensors):
                raise RuntimeError(
                    f"[MegaKittens] Node arity mismatch: out_nodes={len(node.out_nodes)} out_tensors={len(node.out_tensors)}"
                )
            if len(node.in_ranges) != len(node.in_nodes):
                raise RuntimeError(
                    f"[MegaKittens] Node arity mismatch: in_ranges={len(node.in_ranges)} in_nodes={len(node.in_nodes)}"
                )
            if len(node.out_ranges) != len(node.out_tensors):
                raise RuntimeError(
                    f"[MegaKittens] Node arity mismatch: out_ranges={len(node.out_ranges)} out_tensors={len(node.out_tensors)}"
                )
            for label, edges in [("in_node", list(zip(node.in_nodes, node.in_ranges))), ("out_tensor", list(zip(node.out_tensors, node.out_ranges)))]:
                for i, (src, range) in enumerate(edges):
                    src_shape = src[0].out_tensors[src[1]].shape if label == "in_node" else src.shape
                    pad = len(range) - len(src_shape)
                    for d, dim_range in enumerate(range.ranges[:pad]):
                        if dim_range.start != 0 or dim_range.stop != 1 or dim_range.stride != 1:
                            raise RuntimeError(
                                f"[MegaKittens] Range dim {d} ({dim_range.start}, {dim_range.stop}, {dim_range.stride}) "
                                f"is outside tensor shape {src_shape} and must be (0, 1, 1) for {label} {i}"
                            )
                    for d, (dim_range, dim_size) in enumerate(zip(range.ranges[pad:], src_shape)):
                        if dim_range.stop > dim_size:
                            raise RuntimeError(
                                f"[MegaKittens] Range dim {pad + d} stop ({dim_range.stop}) > tensor dim ({dim_size}) "
                                f"for {label} {i}"
                            )

        input_nodes = [node for node in self.nodes if node.is_input]
        output_nodes = [node for node in self.nodes if node.is_output]

        for node in input_nodes:
            if len(node.in_nodes) != 0:
                raise RuntimeError("[MegaKittens] Input node has inbound edges")

        for node in output_nodes:
            if any(len(dst_nodes) != 0 for dst_nodes in node.out_nodes):
                raise RuntimeError("[MegaKittens] Output node has outbound edges")

        if len(output_nodes) != 1:
            raise RuntimeError(f"[MegaKittens] Number of output nodes is {len(output_nodes)}")

        self._validate_topological()
