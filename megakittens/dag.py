from enum import Enum
from typing import Any, Literal, Tuple

from pydantic import BaseModel, NonNegativeInt


class DType(str, Enum):
    fp64 = "fp64"
    fp32 = "fp32"
    bf16 = "bf16"
    half = "half"
    fp8e4m3 = "fp8e4m3"
    fp8e5m2 = "fp8e5m2"
    fp8e8m0 = "fp8e8m0"
    fp4e2m1x2 = "fp4e2m1x2"


class OpType(str, Enum):
    add = "add"
    matmul = "matmul"
    relu = "relu"


class Node(BaseModel):
    dtype: DType
    shape: Tuple[NonNegativeInt, ...] # TODO: support dynamic shapes
    device: Literal["cuda"] # TODO: support other device types
    default: Any

    model_config = {"frozen": True}


class Edge(BaseModel):
    optype: OpType
    in_nodes: Tuple[Node]
    out_nodes: Tuple[Node]

    model_config = {"frozen": True}
