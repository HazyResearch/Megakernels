import math
from typing import List, Tuple

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorSpec


# Page size and count must match csrc config (default_config)
_PAGE_BYTES = 32768  # Config::PAGE_SIZE
_NUM_PAGES = 7       # approximate Config::NUM_PAGES on Blackwell (228KB smem)


def _rows_per_inst(N: int, num_pages: int = _NUM_PAGES) -> int:
    """How many full rows of width N fit in the page buffer (page-aligned)."""
    bytes_per_row = N * 2  # bf16
    if bytes_per_row <= _PAGE_BYTES:
        # Multiple rows per page
        rows_per_page = _PAGE_BYTES // bytes_per_row
        return rows_per_page * num_pages
    else:
        # One row spans multiple pages
        pages_per_row = bytes_per_row // _PAGE_BYTES
        return max(num_pages // pages_per_row, 1)


class RMSNorm(IType):
    @property
    def name(self) -> str:
        return "rmsnorm"

    @property
    def cpp_template(self) -> str:
        return "RMSNorm<MKConfig, MKGlobals, {tensors}>"

    @property
    def cpp_include(self) -> str:
        return "itypes/rmsnorm.cuh"

    @property
    def op_type(self) -> str:
        return "rmsnorm"

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            # x: (M, N) bf16 — no TMA, loaded via cp.async
            TensorSpec(dtype=DType.bf16, granularity=(1, 8)),
            # weight: (N,) bf16 — no TMA, read directly from global
            TensorSpec(dtype=DType.bf16, granularity=(8,)),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            # y: (M, N) bf16 — no TMA, stored via raw global writes
            TensorSpec(dtype=DType.bf16, granularity=(1, 8)),
        ]

    def block_indices(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> List[Tuple[int, ...]]:
        M, N = src_metas[0].shape[-2], src_metas[0].shape[-1]
        rpi = _rows_per_inst(N)
        indices = []
        row = 0
        while row < M:
            rows_this = min(rpi, M - row)
            # (row_start, N, rows_this_instruction)
            indices.append((row, N, rows_this))
            row += rows_this
        return indices

    def num_instructions(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> int:
        M, N = src_metas[0].shape[-2], src_metas[0].shape[-1]
        rpi = _rows_per_inst(N)
        return math.ceil(M / rpi)

    def validate(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> None:
        super().validate(src_metas, dst_metas)
        x_meta = src_metas[0]
        w_meta = src_metas[1]

        if len(x_meta.shape) < 2:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm requires x with at least 2 dims, got shape {x_meta.shape}"
            )

        N = x_meta.shape[-1]

        if w_meta.shape != (N,):
            raise RuntimeError(
                f"[MegaKittens] RMSNorm weight shape {w_meta.shape} doesn't match x last dim {N}"
            )

        if dst_metas[0].shape != x_meta.shape:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm output shape {dst_metas[0].shape} doesn't match input shape {x_meta.shape}"
            )

        # N must be a multiple of 8 (128 elements per warp iteration = 32 threads * 4)
        if N % 8 != 0:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm requires N divisible by 8, got {N}"
            )

        # Must fit at least 1 row in pages
        bytes_per_row = N * 2
        if bytes_per_row > _NUM_PAGES * _PAGE_BYTES:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm row size {bytes_per_row}B exceeds page buffer "
                f"{_NUM_PAGES * _PAGE_BYTES}B. N={N} is too large for single-CTA path."
            )

        # Rows must not cross page boundaries. This holds when:
        # - bytes_per_row evenly divides PAGE_BYTES (multiple rows per page), OR
        # - PAGE_BYTES evenly divides bytes_per_row (one row spans whole pages)
        if _PAGE_BYTES % bytes_per_row != 0 and bytes_per_row % _PAGE_BYTES != 0:
            raise RuntimeError(
                f"[MegaKittens] RMSNorm requires row size ({bytes_per_row}B) to be "
                f"page-aligned (PAGE_SIZE={_PAGE_BYTES}B). N={N} is not supported. "
                f"Standard LLM dimensions (2048, 4096, 8192, 16384) are all supported."
            )
