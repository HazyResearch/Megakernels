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
