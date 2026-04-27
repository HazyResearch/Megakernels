from functools import cached_property
from typing import ClassVar, Literal, Tuple

import torch
from pydantic import BaseModel, Field, NonNegativeInt, conint, field_validator, model_validator

from .device import Device
from .dtype import DType
from ..jit.pykittens import st, sv


class TensorMeta(BaseModel, frozen=True):  # frozen=True needed to be hashable
    dtype: DType
    shape: Tuple[NonNegativeInt, ...] = Field(max_length=4)  # TODO: support dynamic shapes
    device: Device

    @property
    def full_range(self) -> "TensorRange":
        return TensorRange(ranges=tuple(DimRange(start=0, stop=d) for d in self.shape))

    @cached_property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

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


class TensorSlice(BaseModel, frozen=True):
    """One step of a tensor slicing chain.

    - `op="select"`: pick index `start` on `dim` and drop `dim`; `end` must equal
                     `start + 1` (a single-index range). `start` and `end` must not be None.
    - `op="slice"`:  narrow `dim` to the half-open range `[start, end)`; `start=None`
                     means "from the beginning of the current view" and `end=None`
                     means "to the end".

    `dim`/`start`/`end` may be negative and follow Python indexing semantics. `dim`
    always refers to the current view's coordinate system.
    """
    op: Literal["select", "slice"]
    dim: int
    start: int | None = None
    end: int | None = None

    @model_validator(mode='after')
    def _validate(self):
        if self.op == "select" and (self.start is None or self.end is None or self.end != self.start + 1):
            raise ValueError(
                f"[MegaKittens] TensorSlice op='select' requires int start and end == start + 1, "
                f"got start={self.start}, end={self.end}"
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


class TensorRange(BaseModel):
    NUM_DIMS: ClassVar[int] = 4  # ThunderKittens uses 4-dim tensors

    ranges: Tuple[DimRange, ...] = Field(min_length=NUM_DIMS, max_length=NUM_DIMS)

    @field_validator('ranges', mode='before')
    @classmethod
    def _pad_ranges(cls, v):
        v = tuple(v)
        if not 1 <= len(v) <= cls.NUM_DIMS:
            raise ValueError(f"[MegaKittens] TensorRange requires 1 to {cls.NUM_DIMS} DimRanges, got {len(v)}")
        if len(v) < cls.NUM_DIMS:
            v = (DimRange(start=0, stop=1, stride=1),) * (cls.NUM_DIMS - len(v)) + v
        return v

    def __getitem__(self, idx: int) -> DimRange:
        return self.ranges[idx]

    def __len__(self) -> int:
        return len(self.ranges)

    def __iter__(self):
        return iter(self.ranges)

    @property
    def effective_shape(self) -> Tuple[int, ...]:
        return tuple(d.size for d in self.ranges)

    @staticmethod
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

    @classmethod
    def from_slice_chain(cls, shape: Tuple[int, ...], slice_chain: list["TensorSlice"]) -> "TensorRange":
        """Build an absolute `TensorRange` on a tensor of `shape` by replaying `slice_chain`.

        A slice chain is a sequence of `TensorSlice` view ops that successively narrow
        a tensor view. Returns the full range when `slice_chain` is empty.
        """
        if len(shape) > cls.NUM_DIMS:
            raise RuntimeError(f"[MegaKittens] Tensor has {len(shape)} dims, max is {cls.NUM_DIMS}")
        pad = cls.NUM_DIMS - len(shape)
        ranges: list[DimRange] = [DimRange(start=0, stop=1) for _ in range(pad)] + [DimRange(start=0, stop=d) for d in shape]
        view_dims: list[int] = list(range(pad, cls.NUM_DIMS))

        for slice in slice_chain:
            dim = slice.dim
            if dim < 0:  # negative indexing
                dim += len(view_dims)
            if not 0 <= dim < len(view_dims):
                raise RuntimeError(f"[MegaKittens] slice-chain dim {slice.dim} out of range for view with {len(view_dims)} dims")
            target_idx = view_dims[dim]
            prev_range = ranges[target_idx]

            start = 0 if slice.start is None else slice.start
            end = prev_range.size if slice.end is None else slice.end
            if start < 0:
                start += prev_range.size
            if end < 0:
                end += prev_range.size

            if slice.op == "select":
                if not 0 <= start < prev_range.size:
                    raise RuntimeError(f"[MegaKittens] select index {slice.start} out of range [0, {prev_range.size}) at dim {dim}")
                view_dims.pop(dim)  # select collapses this view dim
            elif slice.op == "slice":
                pass  # nothing to be done for slice
            else:
                raise RuntimeError("[MegaKittens] Unexpected slice op")

            # For select op, start == end - 1
            start = max(0, min(start, prev_range.size))
            end = max(start, min(end, prev_range.size))
            if start == end:
                raise RuntimeError(
                    f"[MegaKittens] TensorSlice collapses dim {dim} to empty range, which is not supported"
                    f" (op={slice.op!r}, start={slice.start}, end={slice.end})"
                )
            ranges[target_idx] = DimRange(start=prev_range.start + start, stop=prev_range.start + end)

        return cls(ranges=tuple(ranges))

    def is_full(self, shape: "TensorMeta | Tuple[int, ...]") -> bool:
        """Return True if this range covers the entire tensor. `shape` may be a `TensorMeta` or a raw shape tuple."""
        if isinstance(shape, TensorMeta):
            shape = shape.shape
        if len(shape) > len(self):
            return False
        if any(d.start != 0 or d.stop != 1 or d.stride != 1 for d in self.ranges[:len(self) - len(shape)]):
            return False
        return all(
            d.start == 0 and d.stop == s and d.stride == 1
            for d, s in zip(self.ranges[-len(shape):], shape)
        )
