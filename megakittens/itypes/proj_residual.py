import math
from typing import List, Tuple

import torch

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorSpec
from ..jit.pykittens import sv


@torch.library.custom_op("megakittens::proj_residual", mutates_args=())
def proj_residual_op(
    x: torch.Tensor,
    down_weights: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    return residual + x @ down_weights.T


@proj_residual_op.register_fake
def _proj_residual_fake(
    x: torch.Tensor,
    down_weights: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    M = x.shape[0]
    N = down_weights.shape[0]
    return torch.empty((M, N), dtype=x.dtype, device=x.device)


_PAGE_BYTES = 32768
_OUT_COLS_PER_INST = 16


def _sv_sizeof_bytes(n: int) -> int:
    raw = n * 2
    return ((raw + 127) // 128) * 128


class ProjResidual(IType):

    def __init__(self) -> None:
        self._i = 0

    @property
    def name(self) -> str:
        return "proj_residual"

    @property
    def cpp_template(self) -> str:
        return f"ProjResidual<MKConfig, MKGlobals, {self._i}, {{tensors}}>"

    @property
    def cpp_include(self) -> str:
        return "itypes/proj_residual.cuh"

    @property
    def op_type(self) -> str:
        return "proj_residual"

    @property
    def inputs(self) -> list[TensorSpec]:
        if self._i > 0:
            return [
                TensorSpec(
                    dtype=DType.bf16,
                    granularity=(1, 16),
                    tma_types=[sv(dtype=DType.bf16, length=self._i)],
                ),
                TensorSpec(dtype=DType.bf16, granularity=(1, 16)),
                TensorSpec(dtype=DType.bf16, granularity=(1, 16)),
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1, 16)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 16)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 16)),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1, 16)),
        ]

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ) -> List[Tuple[int, ...]]:
        M = src_metas[0].shape[-2]
        N = src_metas[1].shape[0]
        indices: List[Tuple[int, ...]] = []
        col = 0
        while col < N:
            col_this = min(_OUT_COLS_PER_INST, N - col)
            indices.append((0, M, col, col_this, N))
            col += col_this
        return indices

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ) -> int:
        N = src_metas[1].shape[0]
        return math.ceil(N / _OUT_COLS_PER_INST)

    def validate(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> None:
        x_meta, dw_meta, res_meta = src_metas
        I = x_meta.shape[-1]
        self._i = I

        super().validate(src_metas, dst_metas)

        if len(x_meta.shape) < 2:
            raise RuntimeError(
                f"[MegaKittens] ProjResidual requires x with at least 2 dims, got {x_meta.shape}"
            )
        M = x_meta.shape[-2]
        N = dw_meta.shape[0]

        if len(dw_meta.shape) != 2 or dw_meta.shape[1] != I:
            raise RuntimeError(
                f"[MegaKittens] ProjResidual down_weights must be (N, I), got {dw_meta.shape}"
            )
        if res_meta.shape != (M, N):
            raise RuntimeError(
                f"[MegaKittens] ProjResidual residual shape {res_meta.shape} expected ({M}, {N})"
            )
        if dst_metas[0].shape != (M, N):
            raise RuntimeError(
                f"[MegaKittens] ProjResidual output shape expected ({M}, {N}), got {dst_metas[0].shape}"
            )

        if I % 16 != 0:
            raise RuntimeError(f"[MegaKittens] ProjResidual requires I divisible by 16, got {I}")
        if I > 256 and (I * 2) % 128 != 0:
            raise RuntimeError(
                f"[MegaKittens] ProjResidual requires I <= 256 or I*2 divisible by 128, got {I}"
            )

        row_bytes = _sv_sizeof_bytes(I)
        if row_bytes > _PAGE_BYTES:
            raise RuntimeError(
                f"[MegaKittens] ProjResidual row size {row_bytes}B exceeds page {_PAGE_BYTES}B (I={I})"
            )

        if N % 16 != 0:
            raise RuntimeError(
                f"[MegaKittens] ProjResidual requires N divisible by 16, got {N}"
            )
