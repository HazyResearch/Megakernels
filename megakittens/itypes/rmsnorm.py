import math
from typing import List, Tuple

import torch

from ..dispatcher import Dispatcher
from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorRange, TensorSpec
from ..jit.pykittens import sv


@torch.library.custom_op("megakittens::rmsnorm", mutates_args=())
def rmsnorm_op(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rms_norm(x, [x.shape[-1]], weight, eps)


@rmsnorm_op.register_fake
def _rmsnorm_fake(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)


def _rows_per_inst(C: int) -> int:
    """How many full rows of width C fit in one page."""
    row_bytes = ((C * 2 + 127) // 128) * 128  # sv<bf16, C> padded to 128-byte boundary
    return Dispatcher.PAGE_SIZE // row_bytes


def _resolve_rmsnorm(args, kwargs):
    x_node = args[0]
    col_dim = x_node.meta['val'].shape[-1]
    return RMSNorm(col_dim=col_dim)


def _resolve_aten_fused_rms_norm(args, kwargs):
    x_node = args[0]
    col_dim = x_node.meta['val'].shape[-1]
    return RMSNorm(col_dim=col_dim), [0]


class RMSNorm(IType):
    torch_functions_map = {
        torch.ops.megakittens.rmsnorm: _resolve_rmsnorm,
        torch.ops.megakittens.rmsnorm.default: _resolve_rmsnorm,
        torch.ops.aten._fused_rms_norm: _resolve_aten_fused_rms_norm,
        torch.ops.aten._fused_rms_norm.default: _resolve_aten_fused_rms_norm,
    }
    torch_methods_map = {"rmsnorm": _resolve_rmsnorm}
    torch_modules_map = {torch.nn.RMSNorm: _resolve_rmsnorm}

    test_cases = [
        ((), (1, 64)),
        ((), (1, 2048)), ((), (4, 2048)), ((), (32, 2048)),
        ((), (16, 4096)), ((), (32, 4096)), ((), (8, 8192)),
        ((), (2, 4, 2048)), ((), (2, 3, 4, 2048)),
    ]
    test_atol = 1e-2
    test_rtol = 1e-2
    bench_cases = [((), (32768, 256)), ((), (32768, 512)), ((), (32768, 1536)), ((), (32768, 2048)), ((), (32768, 4096)), ((), (32768, 8192)), ((), (32768, 16384))]

    def test_args(self, case: tuple) -> tuple:
        C = case[-1]
        return (
            torch.randn(*case, dtype=torch.bfloat16, device="cuda"),
            torch.randn(C, dtype=torch.bfloat16, device="cuda"),
            1e-6,
        )

    def bench_bytes(self, case: tuple) -> float:
        num_elements = 1
        for d in case:
            num_elements *= d
        C = case[-1]
        return num_elements * 2 * 2 + C * 2

    def __init__(self, col_dim: int = 0):
        self.col_dim = col_dim

    @property
    def cpp_template(self) -> str:
        return f"RMSNorm<MKConfig, MKGlobals, {self.col_dim}, {{tensors}}>"

    @property
    def inputs(self) -> list[TensorSpec]:
        if self.col_dim > 0:
            return [
                TensorSpec(dtype=DType.bf16, granularity=(1, 64), tma_types=[sv(dtype=DType.bf16, length=self.col_dim)]),
                TensorSpec(dtype=DType.bf16, granularity=(64,)),
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1, 64)),
            TensorSpec(dtype=DType.bf16, granularity=(64,)),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        if self.col_dim > 0:
            return [
                TensorSpec(dtype=DType.bf16, granularity=(1, 64), tma_types=[sv(dtype=DType.bf16, length=self.col_dim)]),
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1, 64)),
        ]

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        x_range = src_ranges[0]
        y_range = dst_ranges[0]
        rows_per_inst = _rows_per_inst(x_range[3].size)
        indices = []
        for b in range(x_range[0].size):
            for d in range(x_range[1].size):
                r = 0
                while r < x_range[2].size:
                    n = min(rows_per_inst, x_range[2].size - r)
                    indices.append((
                        x_range[0].start + b, x_range[1].start + d, x_range[2].start + r,
                        y_range[0].start + b, y_range[1].start + d, y_range[2].start + r,
                        n,
                    ))
                    r += n
        return indices

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> int:
        x_range = src_ranges[0]
        return x_range[0].size * x_range[1].size * math.ceil(x_range[2].size / _rows_per_inst(x_range[3].size))

    def access_regions(
        self,
        block_index: Tuple[int, ...],
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ) -> tuple[list[list[tuple[tuple[int, int], ...]]], list[list[tuple[tuple[int, int], ...]]]]:
        b_x, d_x, r_x, b_y, d_y, r_y, n = block_index
        C = src_metas[0].shape[-1]
        x_region = ((b_x, b_x + 1), (d_x, d_x + 1), (r_x, r_x + n), (0, C))
        w_region = ((0, C),)
        y_region = ((b_y, b_y + 1), (d_y, d_y + 1), (r_y, r_y + n), (0, C))
        return [[x_region], [w_region]], [[y_region]]

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        x_meta = src_metas[0]
        w_meta = src_metas[1]
        C = x_meta.shape[-1]

        if w_meta.shape != (C,):
            raise RuntimeError(
                f"[MegaKittens] RMSNorm weight shape {w_meta.shape} doesn't match x last dim {C}"
            )

        x_range = src_ranges[0]
        w_range = src_ranges[1]
        y_range = dst_ranges[0]

        if x_range[3].start != 0 or x_range[3].stop != C:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm requires full-C range on x, got "
                f"[{x_range[3].start}, {x_range[3].stop}) against C={C}"
            )
        if y_range[3].start != 0 or y_range[3].stop != dst_metas[0].shape[-1]:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm requires full-C range on y, got "
                f"[{y_range[3].start}, {y_range[3].stop}) against C={dst_metas[0].shape[-1]}"
            )
        if not w_range.is_full(w_meta):
            raise RuntimeError(
                f"[MegaKittens] RMSNorm weight must be full-range, got "
                f"[{w_range[0].start}, {w_range[0].stop}) against C={C}"
            )
        if x_range.effective_shape != y_range.effective_shape:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm effective shape mismatch: "
                f"x={x_range.effective_shape} y={y_range.effective_shape}"
            )

        # TMA sv constraint: length <= 256 OR (length * sizeof(dtype)) % 128 == 0
        if C > 256 and (C * 2) % 128 != 0:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm requires C <= 256 or C*2 divisible by 128, got {C}"
            )

        # Row must fit in one page
        row_bytes = ((C * 2 + 127) // 128) * 128
        if row_bytes > Dispatcher.PAGE_SIZE:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm row size {row_bytes}B exceeds page size "
                f"{Dispatcher.PAGE_SIZE}B. C={C} is too large for single-page TMA path."
            )
