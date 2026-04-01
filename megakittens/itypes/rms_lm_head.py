import torch

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorSpec
from ..jit.pykittens import sv, st


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
        return "itypes/llama1b/rms_lm_head.cuh"

    @property
    def op_type(self) -> str:
        return "rms_lm_head"

    @property
    def inputs(self) -> list[TensorSpec]:
        if self._n > 0:
            return [
                # SRC0: hidden_states — TMA sv load into activation page
                TensorSpec(dtype=DType.bf16, granularity=(1,),
                           tma_types=[sv(dtype=DType.bf16, length=self._n)]),
                # SRC1: norm_weight — TMA sv load into activation page
                TensorSpec(dtype=DType.bf16, granularity=(1,),
                           tma_types=[sv(dtype=DType.bf16, length=self._n)]),
                # SRC2: lm_head_weights — TMA st load (16×512 tiles)
                TensorSpec(dtype=DType.bf16, granularity=(1, 1),
                           tma_types=[st(dtype=DType.bf16, rows=16, cols=512)]),
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,)),
            TensorSpec(dtype=DType.bf16, granularity=(1,)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1)),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            # DST: logits — TMA sv store (16-element chunks)
            TensorSpec(dtype=DType.bf16, granularity=(1,),
                       tma_types=[sv(dtype=DType.bf16, length=16)]),
        ]

    def block_indices(self, src_metas, dst_metas):
        return [()]

    def validate(self, src_metas, dst_metas):
        x_meta = src_metas[0]
        self._n = x_meta.shape[-1]
