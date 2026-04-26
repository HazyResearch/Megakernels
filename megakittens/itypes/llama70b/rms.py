from typing import List, Tuple

import torch

from ...dispatcher import Dispatcher
from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorMeta, TensorRange, TensorSpec
from ...jit.pykittens import sv
from ...jit.cuda_utils import get_sm_count


MAX_ROWS_PER_INST = 2 * (Dispatcher.NUM_PAGES - 1)  # page 0 holds weight; remaining pages hold 2 rows each


@torch.library.custom_op("megakittens::rms70b", mutates_args=())
def rms70b_op(x: torch.Tensor, weight: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    return torch.rms_norm(x, [x.shape[-1]], weight, eps.item())


@rms70b_op.register_fake
def _rms70b_fake(x: torch.Tensor, weight: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def _resolve_rms70b(args, kwargs):
    x_node = args[0]
    col_dim = x_node.meta['val'].shape[-1]
    return Rms70b(col_dim=col_dim)


class Rms70b(IType):
    torch_functions_map = {
        torch.ops.megakittens.rms70b: _resolve_rms70b,
        torch.ops.megakittens.rms70b.default: _resolve_rms70b,
    }

    test_cases = [
        ((8192,), (128, 8192)),
        ((8192,), (256, 8192)),
        ((8192,), (512, 8192)),
        ((8192,), (1024, 8192)),
    ]
    test_atol = 1e-2
    test_rtol = 1e-2
    bench_cases = [
        ((8192,), (1024, 8192)),
    ]

    def __init__(self, col_dim: int = 0):
        self.col_dim = col_dim

    @property
    def cpp_template(self) -> str:
        return f"llama70b::RMS<MKConfig, MKGlobals, {self.col_dim}, {{tensors}}>"

    @property
    def cpp_include(self) -> str:
        return "itypes/llama70b/rms.cuh"

    def test_args(self, case: tuple) -> tuple:
        C = case[-1]
        return (
            torch.randn(*case, dtype=torch.bfloat16, device="cuda"),
            torch.randn(C, dtype=torch.bfloat16, device="cuda"),
            torch.tensor([1e-6], dtype=torch.float32, device="cuda"),
        )

    @property
    def inputs(self) -> list[TensorSpec]:
        if self.col_dim > 0:
            return [
                TensorSpec(dtype=DType.bf16, granularity=(1, 64),
                           tma_types=[sv(dtype=DType.bf16, length=self.col_dim)]),
                TensorSpec(dtype=DType.bf16, granularity=(64,),
                           tma_types=[sv(dtype=DType.bf16, length=self.col_dim)]),
                TensorSpec(dtype=DType.fp32, granularity=(1,)),
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1, 64)),
            TensorSpec(dtype=DType.bf16, granularity=(64,)),
            TensorSpec(dtype=DType.fp32, granularity=(1,)),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        if self.col_dim > 0:
            return [
                TensorSpec(dtype=DType.bf16, granularity=(1, 64),
                           tma_types=[sv(dtype=DType.bf16, length=self.col_dim)]),
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1, 64)),
        ]

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> int:
        return min(src_ranges[0][-2].size, get_sm_count())

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        x_range = src_ranges[0]
        w_range = src_ranges[1]
        layer_idx = w_range[-2].start  # 0 for lm_head_norm (1D), layer index for attn/mlp norm (2D)
        B = x_range[-2].size
        sm_count = get_sm_count()
        n_inst = min(B, sm_count)
        return [
            (layer_idx, 0,
             x_range[-2].start + round(i * B / n_inst),
             round((i + 1) * B / n_inst) - round(i * B / n_inst))
            for i in range(n_inst)
        ]

    def access_regions(
        self,
        block_index: Tuple[int, ...],
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ):
        layer_idx, _, row_start, num_rows = block_index
        C = src_metas[0].shape[-1]
        leading_region = tuple((0, 1) for _ in range(len(src_metas[0].shape) - 2))
        x_region = leading_region + ((row_start, row_start + num_rows), (0, C))
        # Weight is either (C,) for lm_head_norm or (NUM_LAYERS, C) for attn/mlp norm.
        # Use layer_idx to point at the correct slice.
        if len(src_metas[1].shape) == 1:
            w_region = ((0, C),)
        else:
            w_region = ((layer_idx, layer_idx + 1), (0, C))
        eps_region = ((0, 1),)
        y_region = leading_region + ((row_start, row_start + num_rows), (0, C))
        return [[x_region], [w_region], [eps_region]], [[y_region]]

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        C = src_metas[0].shape[-1]
        if C != self.col_dim:
            raise RuntimeError(
                f"[MegaKittens] Rms70b expected col_dim={self.col_dim}, got {C}"
            )

        B = src_ranges[0][-2].size
        sm_count = get_sm_count()
        n_inst = min(B, sm_count)
        rows_per_inst = (B + n_inst - 1) // n_inst if n_inst > 0 else 0
        if rows_per_inst > MAX_ROWS_PER_INST:
            raise RuntimeError(
                f"[MegaKittens] Rms70b batch size B={B} gives rows_per_inst={rows_per_inst}, "
                f"exceeds MAX_ROWS_PER_INST={MAX_ROWS_PER_INST}. "
                f"Max supported B for this layout is {MAX_ROWS_PER_INST * sm_count}."
            )
