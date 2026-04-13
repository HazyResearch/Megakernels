from typing import Tuple

import torch
from pydantic import BaseModel, Field, NonNegativeInt, conint, model_validator

from .device import Device
from .dtype import DType
from ..jit.pykittens import st, sv


class TensorMeta(BaseModel, frozen=True):  # frozen=True needed to be hashable
    dtype: DType
    shape: Tuple[NonNegativeInt, ...] = Field(max_length=4)  # TODO: support dynamic shapes
    device: Device

    @classmethod
    def from_torch(
        cls,
        source: torch.Tensor | torch.fx.passes.shape_prop.TensorMetadata,
        fallback_device: torch.device | None = None
    ) -> "TensorMeta":
        if isinstance(source, torch.Tensor):
            if not source.is_contiguous():
                raise RuntimeError(f"[MegaKittens] All tensors must be contiguous")
            return cls(
                dtype=DType.from_torch(source.dtype),
                shape=tuple(int(dim) for dim in source.shape),
                device=Device.from_torch(source.device),
            )
        elif isinstance(source, torch.fx.passes.shape_prop.TensorMetadata):
            if source.stride is not None:
                ndim = len(source.shape)
                if ndim > 0:
                    expected = 1
                    for i in range(ndim - 1, -1, -1):
                        if source.shape[i] != 1 and source.strides[i] != expected:
                            raise RuntimeError(f"[MegaKittens] All tensors must be contiguous")
                        expected *= source.shape[i]
            if source.dtype is None:
                raise ValueError("[MegaKittens] TensorMetadata has no dtype")
            device = getattr(source, "device", None) or fallback_device
            if device is None:
                raise ValueError("[MegaKittens] TensorMetadata has no device and no fallback provided")
            return cls(
                dtype=DType.from_torch(source.dtype),
                shape=tuple(int(dim) for dim in source.shape),
                device=Device.from_torch(device),
            )
        else:
            raise TypeError(f"[MegaKittens] Cannot create TensorMeta from {type(source).__name__}")


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
