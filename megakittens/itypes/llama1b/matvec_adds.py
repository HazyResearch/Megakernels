import torch

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorSpec
from ...jit.pykittens import sv, st


@torch.library.custom_op("megakittens::matvec_adds", mutates_args=())
def matvec_adds_op(
    x: torch.Tensor,
    down_weights: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    return residual + x @ down_weights.T


@matvec_adds_op.register_fake
def _matvec_adds_fake(x, down_weights, residual):
    return torch.empty_like(residual)


BLOCK_SIZE = 16


class MatVecAdds(IType):

    def __init__(self, n=0):
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
                TensorSpec(dtype=DType.bf16, granularity=(1,)),                          # activations
                TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1),                      # weights
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
                TensorSpec(dtype=DType.bf16, granularity=(1,),                           # output (store_add)
                           tma_types=[sv(dtype=DType.bf16, length=16)]),
            ]
        return [TensorSpec(dtype=DType.bf16, granularity=(1,))]

    def block_indices(self, src_metas, dst_metas):
        return [()]

    def validate(self, src_metas, dst_metas):
        if self._n == 0:
            self._n = src_metas[0].shape[-1]
