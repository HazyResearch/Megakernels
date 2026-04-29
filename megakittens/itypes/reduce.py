from typing import List, Tuple

import torch

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorRange, TensorSpec


@torch.library.custom_op("megakittens::reduce", mutates_args=())
def reduce_op(x: torch.Tensor, op: str) -> torch.Tensor:
    if op == "mean":
        return x.mean(dim=-1)
    raise RuntimeError(f"[MegaKittens] Reduce: unknown op {op!r}")


@reduce_op.register_fake
def _reduce_fake(x: torch.Tensor, op: str) -> torch.Tensor:
    if op != "mean":
        raise RuntimeError(f"[MegaKittens] Reduce: unknown op {op!r}")
    return torch.empty(*x.shape[:-1], dtype=x.dtype, device=x.device)


def _detect_dtype(args):
    if args and hasattr(args[0], 'meta') and 'val' in args[0].meta:
        val = args[0].meta['val']
        if isinstance(val, torch.Tensor):
            return DType.from_torch(val.dtype)
    raise RuntimeError(f"[MegaKittens] Reduce: cannot detect dtype from args")


def _resolve_from_custom_op(args, kwargs):
    op = args[1] if len(args) > 1 else kwargs.get("op", "mean")
    return Reduce(op=op, dtype=_detect_dtype(args))


def _resolve_aten_mean_dim(args, kwargs):
    dims = args[1] if len(args) > 1 else kwargs.get("dim", [-1])
    if dims != [-1] and dims != (-1,):
        raise RuntimeError(f"[MegaKittens] Reduce: only dim=[-1] is supported, got {dims}")
    return Reduce(op="mean", dtype=_detect_dtype(args))


class Reduce(IType):
    TILE_ROWS = 128
    REDUCE_GRANULARITY = 16

    REDUCE_OPS = {
        "mean": "ReduceOp::MEAN",
    }

    torch_functions_map = {
        torch.ops.megakittens.reduce: _resolve_from_custom_op,
        torch.ops.megakittens.reduce.default: _resolve_from_custom_op,
        torch.ops.aten.mean.dim: _resolve_aten_mean_dim,
    }

    test_cases = [
        (("mean",), shape)
        for shape in [
            (128, 16),
            (512, 256),
            (1280, 2048),
            (2, 128, 64),
            (3, 512, 256),
            (2, 3, 128, 64),
        ]
    ]
    test_atol = 1e-2
    test_rtol = 1e-2
    bench_cases = [
        (("mean",), (4096, 4096)),
        (("mean",), (131072, 4096)),
        (("mean",), (4096, 131072)),
    ]

    def __init__(self, op: str = "mean", dtype: DType = DType.bf16):
        self.op = op
        self.dtype = dtype

    def test_args(self, case: tuple) -> tuple:
        x = torch.randn(*case, dtype=self.dtype.torch_dtype, device="cuda")
        return (x, self.op)

    def bench_bytes(self, case: tuple) -> float:
        *outer, reduce_dim = case
        rows = 1
        for dim in outer:
            rows *= dim
        return (rows * reduce_dim + rows) * self.dtype.size

    @property
    def cpp_template(self) -> str:
        return f"Reduce<MKConfig, MKGlobals, {self.dtype.cpp_dtype}, {self.REDUCE_OPS[self.op]}, {{tensors}}>"

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=self.dtype, granularity=(self.TILE_ROWS, self.REDUCE_GRANULARITY), tma_types=[]),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=self.dtype, granularity=(self.TILE_ROWS,), tma_types=[]),
        ]

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        src_range = src_ranges[0]
        dst_range = dst_ranges[0]
        indices = []
        for b in range(src_range[0].size):
            for d in range(src_range[1].size):
                for r in range(0, src_range[2].size, self.TILE_ROWS):
                    num_rows = min(self.TILE_ROWS, src_range[2].size - r)
                    indices.append((
                        src_range[0].start + b, src_range[1].start + d, src_range[2].start + r, src_range[3].start,
                        dst_range[0].start, dst_range[1].start + b, dst_range[2].start + d, dst_range[3].start + r,
                        num_rows, src_range[3].size,
                    ))
        return indices

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> int:
        src_range = src_ranges[0]
        return src_range[0].size * src_range[1].size * ((src_range[2].size + self.TILE_ROWS - 1) // self.TILE_ROWS)

    def access_regions(
        self,
        block_index: Tuple[int, ...],
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ) -> tuple[list[list[tuple[tuple[int, int], ...]]], list[list[tuple[tuple[int, int], ...]]]]:
        src_b, src_d, src_r, src_c, dst_b, dst_d, dst_r, dst_c, num_rows, num_cols = block_index
        src_region = (
            (src_b, src_b + 1),
            (src_d, src_d + 1),
            (src_r, src_r + num_rows),
            (src_c, src_c + num_cols),
        )
        dst_region = (
            (dst_b, dst_b + 1),
            (dst_d, dst_d + 1),
            (dst_r, dst_r + 1),
            (dst_c, dst_c + num_rows),
        )
        return [[src_region]], [[dst_region]]

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        if self.op not in self.REDUCE_OPS:
            raise RuntimeError(f"[MegaKittens] Reduce: unknown op {self.op!r}")
        if len(src_metas[0].shape) < 2:
            raise RuntimeError(f"[MegaKittens] Reduce requires an input tensor with at least 2 dimensions")
        src_shape = src_ranges[0].effective_shape
        dst_shape = dst_ranges[0].effective_shape
        if dst_shape[0] != 1 or dst_shape[1:] != src_shape[:3]:
            raise RuntimeError(
                f"[MegaKittens] Reduce effective shape mismatch: "
                f"src={src_shape} dst={dst_shape}; expected dst=(1, *src[:-1])"
            )
