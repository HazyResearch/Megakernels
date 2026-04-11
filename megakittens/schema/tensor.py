from typing import Tuple

from pydantic import BaseModel, Field, NonNegativeInt, conint, model_validator

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
    granularity: tuple[conint(gt=0), ...] = Field(min_length=1)  # each dim must be a multiple of these values
    tma_types: list[st | sv] = []

    @model_validator(mode='after')
    def _validate(self):
        for i, tma in enumerate(self.tma_types):
            if tma.dtype != self.dtype:
                raise ValueError(
                    f"tma_types[{i}].dtype={tma.dtype} doesn't match spec dtype={self.dtype}"
                )
        return self
