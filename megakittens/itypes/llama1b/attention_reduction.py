from typing import List, Tuple

import torch

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorMeta, TensorRange, TensorSpec
from ...jit.pykittens import sv


@torch.library.custom_op("megakittens::attention_reduction", mutates_args=())
def attention_reduction_op(
    lse_intermediates: torch.Tensor, o_intermediates: torch.Tensor,
) -> torch.Tensor:
    # lse_intermediates: [num_heads, num_partials] fp32
    # o_intermediates: [num_heads, num_partials, head_dim] fp32
    # returns: [num_heads * head_dim] bf16
    num_heads, num_partials, head_dim = o_intermediates.shape
    lse = lse_intermediates[:, :num_partials]
    max_lse = lse.max(dim=-1, keepdim=True).values
    weights = torch.exp2(lse - max_lse)
    denom = weights.sum(dim=-1, keepdim=True)
    reduced = (o_intermediates * weights.unsqueeze(-1)).sum(dim=1) / denom
    return reduced.reshape(-1).to(torch.bfloat16)


@attention_reduction_op.register_fake
def _attention_reduction_fake(lse_intermediates, o_intermediates):
    num_heads, _, head_dim = o_intermediates.shape
    return torch.empty(num_heads * head_dim, dtype=torch.bfloat16,
                       device=o_intermediates.device)


class AttentionReduction(IType):

    test_cases = [
        ((), (32, 4, 64)),  # (num_heads, num_partials, head_dim)
    ]
    test_atol = 1e-2
    test_rtol = 1e-2
    bench_cases = []

    def __init__(self, head_dim=64, q_heads_per_instruction=4, max_partials=16):
        self._head_dim = head_dim
        self._q_heads_per_instruction = q_heads_per_instruction
        self._max_partials = max_partials

    @property
    def name(self) -> str:
        return "attention_reduction"

    @property
    def cpp_template(self) -> str:
        return (f"AttentionReduction<MKConfig, MKGlobals, "
                f"{self._head_dim}, {self._q_heads_per_instruction}, "
                f"{self._max_partials}, {{tensors}}>")

    @property
    def cpp_include(self) -> str:
        return "itypes/llama1b/attention_reduction.cuh"

    @property
    def inputs(self) -> list[TensorSpec]:
        lse_padded = ((self._max_partials + 15) // 16) * 16
        return [
            TensorSpec(dtype=DType.fp32, granularity=(self._q_heads_per_instruction, lse_padded),
                       tma_types=[sv(dtype=DType.fp32, length=lse_padded)]),  # lse_intermediates
            TensorSpec(dtype=DType.fp32, granularity=(self._q_heads_per_instruction, 1, self._head_dim),
                       tma_types=[sv(dtype=DType.fp32, length=self._head_dim)]),  # o_intermediates
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self._head_dim,),
                       tma_types=[sv(dtype=DType.bf16, length=self._head_dim)]),  # attn_out
        ]

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> int:
        lse_range = src_ranges[0]
        return lse_range[-2].size // self._q_heads_per_instruction

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        lse_range = src_ranges[0]
        o_range = src_ranges[1]
        num_partials = o_range[-2].size
        return [(0, q_start, num_partials) for q_start in
                range(lse_range[-2].start, lse_range[-2].stop, self._q_heads_per_instruction)]

    def test_args(self, case):
        num_heads, num_partials, head_dim = case
        lse_cols = ((num_partials + 15) // 16) * 16
        lse = torch.zeros(num_heads, lse_cols, dtype=torch.float32, device="cuda")
        lse[:, :num_partials] = torch.randn(num_heads, num_partials, dtype=torch.float32, device="cuda")
        o_partial = torch.randn(num_heads, num_partials, head_dim, dtype=torch.float32, device="cuda")
        return (lse, o_partial)

    def access_regions(self, block_index, src_metas, dst_metas):
        _, q_start, num_partials = block_index
        q_end = q_start + self._q_heads_per_instruction
        head_dim = self._head_dim
        lse_cols = src_metas[0].shape[1]
        lse_region = ((q_start, q_end), (0, lse_cols))
        o_region = ((q_start, q_end), (0, num_partials), (0, head_dim))
        out_region = ((q_start * head_dim, q_end * head_dim),)
        return [[lse_region], [o_region]], [[out_region]]

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)