"""
Pipelined projection + residual add instruction type for LLaMA 1B decode.
Used for both o_proj and down_proj.

Computes: output += weights[layer][rows, reduction_slice] @ input[reduction_slice]

Uses the llama1b matvec pipeline for overlapped weight loading + computation,
and TMA store_add_async for atomic residual accumulation.
"""

from typing import List, Tuple

import torch

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorSpec
from ..jit.pykittens import sv, st


@torch.library.custom_op("megakittens::matvec_adds", mutates_args=())
def matvec_adds_op(
    x: torch.Tensor,
    down_weights: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    return residual + x @ down_weights.T


@matvec_adds_op.register_fake
def _matvec_adds_fake(
    x: torch.Tensor,
    down_weights: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


BLOCK_SIZE = 16


class MatVecAdds(IType):

    def __init__(self, n: int = 0) -> None:
        self._n = n

    @property
    def name(self) -> str:
        return "matvec_adds"

    @property
    def cpp_template(self) -> str:
        return f"MatVecAdds<MKConfig, MKGlobals, {self._n}, {{tensors}}>"

    @property
    def cpp_include(self) -> str:
        return "itypes/llama1b/matvec_adds.cuh"

    @property
    def op_type(self) -> str:
        return "matvec_adds"

    @property
    def inputs(self) -> list[TensorSpec]:
        if self._n > 0:
            return [
                # SRC0: input activations — warp::load from global (no TMA needed)
                TensorSpec(dtype=DType.bf16, granularity=(1,)),
                # SRC1: weights [layers, out_dim, reduction_dim] — TMA st tiles
                TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1),
                           tma_types=[st(dtype=DType.bf16, rows=16, cols=512)]),
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1)),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        if self._n > 0:
            return [
                # DST: output vector — TMA store_add_async (16-element sv chunks)
                TensorSpec(dtype=DType.bf16, granularity=(1,),
                           tma_types=[sv(dtype=DType.bf16, length=16)]),
            ]
        return [TensorSpec(dtype=DType.bf16, granularity=(1,))]

    def block_indices(
        self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...],
    ) -> List[Tuple[int, ...]]:
        return [()]

    def validate(
        self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...],
    ) -> None:
        if self._n == 0:
            self._n = src_metas[0].shape[-1]
