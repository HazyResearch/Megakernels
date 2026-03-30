import math
from typing import List, Tuple

import torch

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorSpec
from ..jit.pykittens import sv


@torch.library.custom_op("megakittens::rms_lm_head", mutates_args=())
def rms_lm_head_op(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    h = torch.rms_norm(x, [x.shape[-1]], norm_weight, eps)
    return h @ lm_head.T


@rms_lm_head_op.register_fake
def _rms_lm_head_fake(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    V = lm_head.shape[0]
    M = x.shape[0]
    return torch.empty((M, V), dtype=x.dtype, device=x.device)


_PAGE_BYTES = 32768
_VOCAB_ROWS_PER_INST = 128


def _sv_sizeof_bytes(n: int) -> int:
    raw = n * 2
    return ((raw + 127) // 128) * 128


def _rows_per_inst(N: int) -> int:
    row_bytes = _sv_sizeof_bytes(N)
    return _PAGE_BYTES // row_bytes


class RmsLmHead(IType):

    def __init__(self) -> None:
        self._n = 0

    @property
    def name(self) -> str:
        return "rms_lm_head"

    @property
    def cpp_template(self) -> str:
        return f"RmsLmHead<MKConfig, MKGlobals, {self._n}, {{tensors}}>"

    @property
    def cpp_include(self) -> str:
        return "itypes/rms_lm_head.cuh"

    @property
    def op_type(self) -> str:
        return "rms_lm_head"

    @property
    def inputs(self) -> list[TensorSpec]:
        if self._n > 0:
            return [
                TensorSpec(
                    dtype=DType.bf16,
                    granularity=(1, 16),
                    tma_types=[sv(dtype=DType.bf16, length=self._n)],
                ),
                TensorSpec(dtype=DType.bf16, granularity=(16,)),
                TensorSpec(dtype=DType.bf16, granularity=(1, 16)),
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1, 16)),
            TensorSpec(dtype=DType.bf16, granularity=(16,)),
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
        M, N = src_metas[0].shape[-2], src_metas[0].shape[-1]
        V = src_metas[2].shape[0]
        indices: List[Tuple[int, ...]] = []
        vocab = 0
        while vocab < V:
            vocab_this = min(_VOCAB_ROWS_PER_INST, V - vocab)
            indices.append((0, M, vocab, vocab_this, V))
            vocab += vocab_this
        return indices

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ) -> int:
        V = src_metas[2].shape[0]
        return math.ceil(V / _VOCAB_ROWS_PER_INST)

    def validate(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> None:
        x_meta, nw_meta, lh_meta = src_metas
        N = x_meta.shape[-1]
        self._n = N

        super().validate(src_metas, dst_metas)

        if len(x_meta.shape) < 2:
            raise RuntimeError(
                f"[MegaKittens] RmsLmHead requires x with at least 2 dims, got {x_meta.shape}"
            )
        M = x_meta.shape[-2]
        if nw_meta.shape != (N,):
            raise RuntimeError(
                f"[MegaKittens] RmsLmHead norm weight shape {nw_meta.shape} expected ({N},)"
            )
        if len(lh_meta.shape) != 2 or lh_meta.shape[1] != N:
            raise RuntimeError(
                f"[MegaKittens] RmsLmHead lm_head must be (V, N), got {lh_meta.shape}"
            )
        V = lh_meta.shape[0]
        if dst_metas[0].shape != (M, V):
            raise RuntimeError(
                f"[MegaKittens] RmsLmHead logits shape expected ({M}, {V}), got {dst_metas[0].shape}"
            )

        if N % 16 != 0:
            raise RuntimeError(f"[MegaKittens] RmsLmHead requires N divisible by 16, got {N}")
        if N > 256 and (N * 2) % 128 != 0:
            raise RuntimeError(
                f"[MegaKittens] RmsLmHead requires N <= 256 or N*2 divisible by 128, got {N}"
            )

        row_bytes = _sv_sizeof_bytes(N)
        if row_bytes > _PAGE_BYTES:
            raise RuntimeError(
                f"[MegaKittens] RmsLmHead row size {row_bytes}B exceeds page {_PAGE_BYTES}B (N={N})"
            )

        if V % 16 != 0:
            raise RuntimeError(
                f"[MegaKittens] RmsLmHead requires V divisible by 16 for logits layout, got {V}"
            )
