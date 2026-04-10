
import math
from typing import List, Tuple

import torch

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorSpec
from ..jit.pykittens import sv


@torch.library.custom_op("megakittens::rmsnorm", mutates_args=())
def rmsnorm_op(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rms_norm(x, [x.shape[-1]], weight, eps)

@rmsnorm_op.register_fake
def _rmsnorm_fake(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)


_PAGE_BYTES = 32768  # Config::PAGE_SIZE


def _sv_sizeof_bytes(n: int) -> int:
    """sizeof(sv<bf16, N>) on Blackwell -- padded to 128-byte boundary."""
    raw = n * 2  # bf16 = 2 bytes
    return ((raw + 127) // 128) * 128


def _rows_per_inst(N: int) -> int:
    """How many full rows of width N fit in one page."""
    row_bytes = _sv_sizeof_bytes(N)
    return _PAGE_BYTES // row_bytes


class RMSNorm(IType):
    torch_functions = []
    torch_methods = ["rmsnorm"]
    torch_modules = [torch.nn.RMSNorm]

    test_cases = [(1, 2048), (4, 2048), (32, 2048), (16, 4096), (32, 4096), (8, 8192)]
    test_atol = 1e-2
    test_rtol = 1e-2
    bench_shapes = [(32768, 256), (32768, 512), (32768, 1536), (32768, 2048), (32768, 4096), (32768, 8192), (32768, 16384)]

    def test_args(self, shape: tuple) -> tuple:
        M, N = shape
        return (
            torch.randn(M, N, dtype=torch.bfloat16, device="cuda"),
            torch.randn(N, dtype=torch.bfloat16, device="cuda"),
            1e-6,
        )

    def bench_bytes(self, shape: tuple) -> float:
        M, N = shape
        return M * N * 2 * 2 + N * 2

    def __init__(self):
        self._n = 0

    @property
    def cpp_template(self) -> str:
        return f"RMSNorm<MKConfig, MKGlobals, {self._n}, {{tensors}}>"

    @property
    def inputs(self) -> list[TensorSpec]:
        if self._n > 0:
            return [
                TensorSpec(dtype=DType.bf16, granularity=(1, 16), tma_types=[sv(dtype=DType.bf16, length=self._n)]),
                TensorSpec(dtype=DType.bf16, granularity=(16,)),
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1, 16)),
            TensorSpec(dtype=DType.bf16, granularity=(16,)),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        if self._n > 0:
            return [
                TensorSpec(dtype=DType.bf16, granularity=(1, 16), tma_types=[sv(dtype=DType.bf16, length=self._n)]),
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1, 16)),
        ]

    def block_indices(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> List[Tuple[int, ...]]:
        M, N = src_metas[0].shape[-2], src_metas[0].shape[-1]
        rpi = _rows_per_inst(N)
        indices = []
        row = 0
        while row < M:
            rows_this = min(rpi, M - row)
            indices.append((row, rows_this))
            row += rows_this
        return indices

    def num_instructions(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> int:
        M, N = src_metas[0].shape[-2], src_metas[0].shape[-1]
        rpi = _rows_per_inst(N)
        return math.ceil(M / rpi)

    def validate(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> None:
        x_meta = src_metas[0]
        w_meta = src_metas[1]
        N = x_meta.shape[-1]

        # Set N for TMA type generation (used by inputs/outputs/cpp_template)
        self._n = N

        super().validate(src_metas, dst_metas)

        if len(x_meta.shape) < 2:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm requires x with at least 2 dims, got shape {x_meta.shape}"
            )

        if w_meta.shape != (N,):
            raise RuntimeError(
                f"[MegaKittens] RMSNorm weight shape {w_meta.shape} doesn't match x last dim {N}"
            )

        if dst_metas[0].shape != x_meta.shape:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm output shape {dst_metas[0].shape} doesn't match input shape {x_meta.shape}"
            )

        # sv requires length divisible by 16
        if N % 16 != 0:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm requires N divisible by 16, got {N}"
            )

        # TMA sv constraint: length <= 256 OR (length * sizeof(dtype)) % 128 == 0
        if N > 256 and (N * 2) % 128 != 0:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm requires N <= 256 or N*2 divisible by 128, got {N}"
            )

        # Row must fit in one page
        row_bytes = _sv_sizeof_bytes(N)
        if row_bytes > _PAGE_BYTES:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm row size {row_bytes}B exceeds page size "
                f"{_PAGE_BYTES}B. N={N} is too large for single-page TMA path."
            )
