from typing import List, Tuple

import torch

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorMeta, TensorSpec
from ...jit.pykittens import st, sv


@torch.library.custom_op("megakittens::rope_gather", mutates_args=())
def rope_gather_op(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor
) -> torch.Tensor:
    # Reference impl: per-token gather of cos/sin by position_ids, then RoPE.
    pos = position_ids.to(torch.long)
    cos_gathered = cos[pos]  # (N, D)
    sin_gathered = sin[pos]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    return x * cos_gathered + rotated * sin_gathered


@rope_gather_op.register_fake
def _fake(x, cos, sin, position_ids):
    return torch.empty_like(x)


class RopeGather(IType):
    HEAD_DIM = 128
    TOKENS_PER_INST = 128

    X_TMA = st(dtype=DType.bf16, rows=128, cols=128)
    Y_TMA = st(dtype=DType.bf16, rows=128, cols=128)
    # cos/sin/position_ids are accessed via warp::load from globals (no TMA
    # descriptors needed); the kernel gathers one row at a time by position_id.

    # (N, D, max_seq_len). Smallest valid shape first per CLAUDE.md.
    test_cases = [((), (128, 128, 128))]
    bench_cases: list[tuple] = []
    test_atol = 1e-2
    test_rtol = 1e-2

    def test_args(self, case: tuple) -> tuple[torch.Tensor, ...]:
        N, D, max_seq = case
        # Build cos/sin tables in LLaMA broadcast layout (cos[:, 2k]==cos[:, 2k+1]).
        positions = torch.arange(max_seq, device="cuda", dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, device="cuda", dtype=torch.float32) / D))
        freqs = positions[:, None] * inv_freq[None, :]  # (max_seq, D/2)
        cos_half = freqs.cos()
        sin_half = freqs.sin()
        cos = torch.stack((cos_half, cos_half), dim=-1).flatten(-2).to(torch.bfloat16)
        sin = torch.stack((sin_half, sin_half), dim=-1).flatten(-2).to(torch.bfloat16)
        x = torch.randn(N, D, dtype=torch.bfloat16, device="cuda")
        position_ids = torch.randint(0, max_seq, (N,), dtype=torch.int32, device="cuda")
        return (x, cos, sin, position_ids)

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.TOKENS_PER_INST, self.HEAD_DIM), tma_types=[self.X_TMA]),
            TensorSpec(dtype=DType.bf16, granularity=(16, self.HEAD_DIM), tma_types=[]),
            TensorSpec(dtype=DType.bf16, granularity=(16, self.HEAD_DIM), tma_types=[]),
            TensorSpec(dtype=DType.int32, granularity=(self.TOKENS_PER_INST,), tma_types=[]),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.TOKENS_PER_INST, self.HEAD_DIM), tma_types=[self.Y_TMA]),
        ]

    def block_indices(self, src_metas, dst_metas, src_ranges, dst_ranges):
        N, D = dst_metas[0].shape
        n_row_tiles = N // self.TOKENS_PER_INST
        n_col_tiles = D // self.HEAD_DIM
        indices = []
        for r in range(n_row_tiles):
            for c in range(n_col_tiles):
                indices.append((r, c))
        return indices

    def num_instructions(self, src_metas, dst_metas, src_ranges, dst_ranges):
        N, D = dst_metas[0].shape
        return (N // self.TOKENS_PER_INST) * (D // self.HEAD_DIM)

    def access_regions(self, block_index, src_metas, dst_metas):
        src_regions = [[tuple((0, s) for s in m.shape)] for m in src_metas]
        dst_regions = [[tuple((0, s) for s in m.shape)] for m in dst_metas]
        return src_regions, dst_regions

    def validate(self, src_metas, dst_metas, src_ranges, dst_ranges):
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        x_shape = src_metas[0].shape
        cos_shape = src_metas[1].shape
        sin_shape = src_metas[2].shape
        pos_shape = src_metas[3].shape
        y_shape = dst_metas[0].shape
        if y_shape != x_shape:
            raise RuntimeError(f"[MegaKittens] RopeGather output shape must match x: x={x_shape}, y={y_shape}")
        if cos_shape != sin_shape:
            raise RuntimeError(f"[MegaKittens] RopeGather cos/sin shapes must match: {cos_shape} vs {sin_shape}")
        if cos_shape[-1] != x_shape[-1]:
            raise RuntimeError(f"[MegaKittens] RopeGather cos last dim must equal x last dim: {cos_shape[-1]} vs {x_shape[-1]}")
        if pos_shape != (x_shape[0],):
            raise RuntimeError(f"[MegaKittens] RopeGather position_ids shape must be (N,): got {pos_shape}, expected ({x_shape[0]},)")
        if x_shape[-1] != self.HEAD_DIM:
            raise RuntimeError(f"[MegaKittens] RopeGather expects last dim == {self.HEAD_DIM}, got {x_shape[-1]}")
