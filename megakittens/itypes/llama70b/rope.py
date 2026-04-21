from typing import List, Tuple

import torch

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorMeta, TensorSpec
from ...jit.pykittens import st


@torch.library.custom_op("megakittens::rope", mutates_args=())
def rope_op(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Matches the CUDA kernel's convention: cos/sin are already per-row
    # (gathered at position_ids by the caller) and use LLaMA's broadcast layout
    # where cos[n, 2k] == cos[n, 2k+1]. The rotation treats adjacent
    # (2k, 2k+1) pairs: rotated = (-x[...,1::2], x[...,::2]) interleaved.
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    return x * cos + rotated * sin


@rope_op.register_fake
def _rope_fake(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class Rope(IType):
    HEAD_DIM = 128
    TOKENS_PER_INST = 128

    TMA = st(dtype=DType.bf16, rows=128, cols=128)

    test_cases = [
        ((), (128, 128)),
        ((), (256, 128)),
        ((), (512, 128)),
        ((), (1024, 128)),
    ]
    bench_cases = [
        ((), (4096, 128)),
        ((), (16384, 128)),
        ((), (65536, 128)),
    ]
    test_atol = 1e-2
    test_rtol = 1e-2

    def test_args(self, case: tuple) -> tuple[torch.Tensor, ...]:
        N, D = case
        # Build cos/sin in the broadcast layout that the CUDA kernel expects.
        # theta_k depends only on the pair index; cos[n, 2k] == cos[n, 2k+1].
        positions = torch.arange(N, device="cuda", dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, device="cuda", dtype=torch.float32) / D))
        freqs = positions[:, None] * inv_freq[None, :]  # (N, D/2)
        cos_half = freqs.cos()
        sin_half = freqs.sin()
        # interleave so col 2k and 2k+1 share the same value
        cos = torch.stack((cos_half, cos_half), dim=-1).flatten(-2).to(torch.bfloat16)
        sin = torch.stack((sin_half, sin_half), dim=-1).flatten(-2).to(torch.bfloat16)
        x = torch.randn(N, D, dtype=torch.bfloat16, device="cuda")
        return (x, cos, sin)

    def bench_bytes(self, case: tuple) -> float:
        N, D = case
        # Read x, cos, sin; write y. Four tensors of bf16 = 2 bytes each.
        return N * D * 2 * 4

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.TOKENS_PER_INST, self.HEAD_DIM), tma_types=[self.TMA]),
            TensorSpec(dtype=DType.bf16, granularity=(self.TOKENS_PER_INST, self.HEAD_DIM), tma_types=[self.TMA]),
            TensorSpec(dtype=DType.bf16, granularity=(self.TOKENS_PER_INST, self.HEAD_DIM), tma_types=[self.TMA]),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.TOKENS_PER_INST, self.HEAD_DIM), tma_types=[self.TMA]),
        ]

    def block_indices(self, src_metas, dst_metas, src_ranges, dst_ranges) -> List[Tuple[int, ...]]:
        N, D = dst_metas[0].shape
        n_row_tiles = N // self.TOKENS_PER_INST
        n_col_tiles = D // self.HEAD_DIM
        indices = []
        for row in range(n_row_tiles):
            for col in range(n_col_tiles):
                indices.append((row, col))
        return indices

    def num_instructions(self, src_metas, dst_metas, src_ranges, dst_ranges) -> int:
        N, D = dst_metas[0].shape
        return (N // self.TOKENS_PER_INST) * (D // self.HEAD_DIM)

    def access_regions(self, block_index, src_metas, dst_metas):
        src_regions = [[tuple((0, s) for s in m.shape)] for m in src_metas]
        dst_regions = [[tuple((0, s) for s in m.shape)] for m in dst_metas]
        return src_regions, dst_regions

    def validate(self, src_metas, dst_metas, src_ranges, dst_ranges) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        x_shape = src_metas[0].shape
        cos_shape = src_metas[1].shape
        sin_shape = src_metas[2].shape
        y_shape = dst_metas[0].shape
        if cos_shape != x_shape or sin_shape != x_shape:
            raise RuntimeError(
                f"[MegaKittens] Rope requires x/cos/sin to share shape; got {x_shape}, {cos_shape}, {sin_shape}"
            )
        if y_shape != x_shape:
            raise RuntimeError(
                f"[MegaKittens] Rope output shape must match x: x={x_shape}, y={y_shape}"
            )
        if x_shape[-1] != self.HEAD_DIM:
            raise RuntimeError(
                f"[MegaKittens] Rope expects last dim == {self.HEAD_DIM}, got {x_shape[-1]}"
            )
