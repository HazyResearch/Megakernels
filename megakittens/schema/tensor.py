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


class DimRange(BaseModel, frozen=True):
    start: NonNegativeInt
    stop: NonNegativeInt  # exclusive
    stride: conint(gt=0) = 1

    @model_validator(mode='after')
    def _validate(self):
        if self.stop < self.start:
            raise ValueError(f"[MegaKittens] DimRange stop ({self.stop}) < start ({self.start})")
        return self

    @property
    def size(self) -> int:
        if self.stride == 1:
            return self.stop - self.start
        return -(-( self.stop - self.start) // self.stride)  # ceiling division


class TensorRange(BaseModel, frozen=True):
    ranges: Tuple[DimRange, ...]

    def __getitem__(self, idx: int) -> DimRange:
        return self.ranges[idx]

    def __len__(self) -> int:
        return len(self.ranges)

    @property
    def effective_shape(self) -> Tuple[int, ...]:
        return tuple(d.size for d in self.ranges)

    @classmethod
    def compose(first: "TensorRange", second: "TensorRange") -> "TensorRange":
        """Compose two ranges: first is applied first (closer to source tensor), 
           second is relative to first's effective_shape."""
        if len(first) != len(second):
            raise ValueError("[MegaKittens] Cannot compose ranges with different ndim")
        new_ranges = []
        for first_range, second_range in zip(first.ranges, second.ranges):
            if first_range.stride != 1 or second_range.stride != 1:
                raise ValueError("[MegaKittens] Cannot compose ranges with stride != 1")
            new_start = first_range.start + second_range.start
            new_stop = first_range.start + second_range.stop
            if new_stop > first_range.stop:
                raise ValueError(
                    f"[MegaKittens] Composed range exceeds parent: "
                    f"parent=[{first_range.start},{first_range.stop}), child=[{second_range.start},{second_range.stop})"
                )
            new_ranges.append(DimRange(start=new_start, stop=new_stop))
        return TensorRange(ranges=tuple(new_ranges))

    def is_full(self, shape: Tuple[int, ...]) -> bool:
        """Return True if this range covers the entire tensor."""
        if len(self) != len(shape):
            return False
        return all(
            d.start == 0 and d.stop == s and d.stride == 1
            for d, s in zip(self.ranges, shape)
        )
