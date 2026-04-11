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
    torch_methods_map = {"rmsnorm": None}
    torch_modules_map = {torch.nn.RMSNorm: None}

    test_cases = [
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
        shape = src_metas[0].shape
        B, D, R, C = (1,) * (4 - len(shape)) + shape
        rows_per_inst = _rows_per_inst(C)
        indices = []
        for b in range(B):
            for d in range(D):
                r = 0
                while r < R:
                    n = min(rows_per_inst, R - r)
                    indices.append((b, d, r, n))
                    r += n
        return indices

    def num_instructions(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> int:
        shape = src_metas[0].shape
        B, D, R, C = (1,) * (4 - len(shape)) + shape
        return B * D * math.ceil(R / _rows_per_inst(C))

    def access_regions(self, block_index, src_metas, dst_metas):
        b, d, r, n = block_index
        C = src_metas[0].shape[-1]
        x_region = ((b, b + 1), (d, d + 1), (r, r + n), (0, C))
        w_region = ((0, C),)
        y_region = ((b, b + 1), (d, d + 1), (r, r + n), (0, C))
        return [x_region, w_region], [y_region]

    def validate(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> None:
        x_meta = src_metas[0]
        w_meta = src_metas[1]
        C = x_meta.shape[-1]

        # Set C for TMA type generation (used by inputs/outputs/cpp_template)
        self._n = C

        super().validate(src_metas, dst_metas)

        if len(x_meta.shape) < 2:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm requires x with at least 2 dims, got shape {x_meta.shape}"
            )

        if w_meta.shape != (C,):
            raise RuntimeError(
                f"[MegaKittens] RMSNorm weight shape {w_meta.shape} doesn't match x last dim {C}"
            )

        if dst_metas[0].shape != x_meta.shape:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm output shape {dst_metas[0].shape} doesn't match input shape {x_meta.shape}"
            )

        # sv requires length divisible by 16
        if C % 16 != 0:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm requires C divisible by 16, got {C}"
            )

        # TMA sv constraint: length <= 256 OR (length * sizeof(dtype)) % 128 == 0
        if C > 256 and (C * 2) % 128 != 0:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm requires C <= 256 or C*2 divisible by 128, got {C}"
            )

        # Row must fit in one page
        row_bytes = _sv_sizeof_bytes(C)
        if row_bytes > _PAGE_BYTES:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm row size {row_bytes}B exceeds page size "
                f"{_PAGE_BYTES}B. C={C} is too large for single-page TMA path."
            )
