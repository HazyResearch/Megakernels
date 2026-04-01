from typing import Tuple

from pydantic import BaseModel, NonNegativeInt

from .device import Device
from .dtype import DType
from ..jit.pykittens import st, sv


class TensorMeta(BaseModel, frozen=True):  # frozen=True needed to be hashable
    dtype: DType
    shape: Tuple[NonNegativeInt, ...]  # TODO: support dynamic shapes
    device: Device


class TensorSpec(BaseModel):
    """Specification for one input or output tensor of an instruction."""
    model_config = {"arbitrary_types_allowed": True}
    dtype: DType
    granularity: tuple[int, ...]  # each dim must be a multiple of these values
    tma_types: list[st | sv] = []
