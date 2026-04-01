import torch

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorSpec
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


class RmsLmHead(IType):

    def __init__(self, n: int = 0) -> None:
        self._n = n

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
                TensorSpec(dtype=DType.bf16, granularity=(1,),
                           tma_types=[sv(dtype=DType.bf16, length=self._n)]),
                TensorSpec(dtype=DType.bf16, granularity=(1,)),
                TensorSpec(dtype=DType.bf16, granularity=(1, 1)),
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,)),
            TensorSpec(dtype=DType.bf16, granularity=(1,)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1)),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [TensorSpec(dtype=DType.bf16, granularity=(1,))]

    def block_indices(self, src_metas, dst_metas):
        return [()]

    def validate(self, src_metas, dst_metas):
        x_meta = src_metas[0]
        self._n = x_meta.shape[-1]
