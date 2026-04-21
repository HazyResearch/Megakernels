"""PGL-aware RMSNorm — the building block for AttnNorm, MlpNorm, LM_HeadNorm.

Mirrors the reference `batched_rms_norm.cu::rms_op<>` template: input and
output are PGL-sharded across devices. Each device's kernel reads its local
slice (`input_pgl.gls[dev_idx]`), computes RMSNorm, writes to its local
output slice. Weight is a regular (replicated) gl.

The reference instantiates the same `rms_op<>` three times for three
logical roles (attn_norm, mlp_norm, lm_head_norm); in our framework those
are all uses of this single itype with different PGL tensor slots.
"""

from typing import List, Tuple

import torch

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorMeta, TensorSpec
from ...jit.pykittens import st, sv


_PGL_NUM_DEVICES_DEFAULT = 8


@torch.library.custom_op("megakittens::pgl_rmsnorm", mutates_args=())
def pgl_rmsnorm_op(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Eager impl matches the local per-device computation. Multi-device
    # semantics are expressed through the PGL-flagged input/output specs.
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    return (x.float() * rstd * weight.float()).to(x.dtype)


@pgl_rmsnorm_op.register_fake
def _fake(x, weight, eps=1e-6):
    return torch.empty_like(x)


class PglRMSNorm(IType):
    """RMSNorm where input and output are PGL-sharded across devices."""

    NUM_DEVICES = _PGL_NUM_DEVICES_DEFAULT

    def __init__(self, n: int = 0):
        # N (hidden dim) is baked as a compile-time C++ template arg.
        # Set during `validate()` from the input tensor's last dim.
        self._n = n

    @property
    def cpp_template(self) -> str:
        return f"PglRMSNorm<MKConfig, MKGlobals, {self._n}, {{tensors}}>"

    @property
    def cpp_include(self) -> str:
        return "itypes/llama70b/pgl_rms_norm.cuh"

    test_cases: list[tuple] = []
    bench_cases: list[tuple] = []
    test_atol = 1e-2
    test_rtol = 1e-2

    def test_args(self, case: tuple) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError("PglRMSNorm is multi-device only; see tests/test_pgl_rms_norm.py")

    @property
    def inputs(self) -> list[TensorSpec]:
        if self._n > 0:
            x_tma = sv(dtype=DType.bf16, length=self._n)
            return [
                TensorSpec(dtype=DType.bf16, granularity=(1, self._n), tma_types=[x_tma],
                           num_devices=self.NUM_DEVICES),
                TensorSpec(dtype=DType.bf16, granularity=(self._n,), tma_types=[]),
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1, 1), tma_types=[], num_devices=self.NUM_DEVICES),
            TensorSpec(dtype=DType.bf16, granularity=(1,), tma_types=[]),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        if self._n > 0:
            y_tma = sv(dtype=DType.bf16, length=self._n)
            return [
                TensorSpec(dtype=DType.bf16, granularity=(1, self._n), tma_types=[y_tma],
                           num_devices=self.NUM_DEVICES),
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1, 1), tma_types=[], num_devices=self.NUM_DEVICES),
        ]

    def block_indices(self, src_metas, dst_metas, src_ranges, dst_ranges) -> List[Tuple[int, ...]]:
        M, N = dst_metas[0].shape
        PAGE_BYTES = 32768
        row_bytes = N * 2  # bf16
        rpi = max(1, PAGE_BYTES // row_bytes)
        indices = []
        row = 0
        while row < M:
            rows_this = min(rpi, M - row)
            indices.append((row, rows_this))
            row += rows_this
        return indices

    def num_instructions(self, src_metas, dst_metas, src_ranges, dst_ranges) -> int:
        return len(self.block_indices(src_metas, dst_metas, src_ranges, dst_ranges))

    def access_regions(self, block_index, src_metas, dst_metas):
        src_regions = [[tuple((0, s) for s in m.shape)] for m in src_metas]
        dst_regions = [[tuple((0, s) for s in m.shape)] for m in dst_metas]
        return src_regions, dst_regions

    def validate(self, src_metas, dst_metas, src_ranges, dst_ranges) -> None:
        x_shape = src_metas[0].shape
        N = x_shape[-1]
        self._n = N
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        if N % 16 != 0:
            raise RuntimeError(f"[MegaKittens] PglRMSNorm N={N} must be a multiple of 16")
        if src_metas[1].shape != (N,):
            raise RuntimeError(f"[MegaKittens] PglRMSNorm weight shape must be (N,)={N}, got {src_metas[1].shape}")
        if dst_metas[0].shape != x_shape:
            raise RuntimeError(f"[MegaKittens] PglRMSNorm output shape must match x: {x_shape} vs {dst_metas[0].shape}")
