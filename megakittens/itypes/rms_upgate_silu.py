import torch

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorSpec
from ..jit.pykittens import sv, st


@torch.library.custom_op("megakittens::rms_upgate_silu", mutates_args=())
def rms_upgate_silu_op(
    x: torch.Tensor, norm_weight: torch.Tensor,
    gate_weight: torch.Tensor, up_weight: torch.Tensor, eps: float,
) -> torch.Tensor:
    h = torch.rms_norm(x, [x.shape[-1]], norm_weight, eps)
    return torch.nn.functional.silu(h @ gate_weight.T) * (h @ up_weight.T)


@rms_upgate_silu_op.register_fake
def _fake(x, norm_weight, gate_weight, up_weight, eps):
    return torch.empty((*x.shape[:-1], gate_weight.shape[0]), dtype=x.dtype, device=x.device)


class RmsUpgateSilu(IType):

    def __init__(self, n=0):
        self._n = n

    @property
    def name(self) -> str:
        return "rms_upgate_silu"

    @property
    def cpp_template(self) -> str:
        return f"RmsUpgateSilu<MKConfig, MKGlobals, {self._n}, {{tensors}}>"

    @property
    def cpp_include(self) -> str:
        return "itypes/llama1b/upgate.cuh"

    @property
    def op_type(self) -> str:
        return "rms_upgate_silu"

    @property
    def inputs(self) -> list[TensorSpec]:
        if self._n > 0:
            return [
                TensorSpec(dtype=DType.bf16, granularity=(1,),                           # hidden_states
                           tma_types=[sv(dtype=DType.bf16, length=self._n)]),
                TensorSpec(dtype=DType.bf16, granularity=(1, 1),                         # norm_weight
                           tma_types=[sv(dtype=DType.bf16, length=self._n)]),
                TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1),                      # up_weights
                           tma_types=[st(dtype=DType.bf16, rows=16, cols=512)]),
                TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1),                      # gate_weights
                           tma_types=[st(dtype=DType.bf16, rows=16, cols=512)]),
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1)),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,),                               # silu_out
                       tma_types=[sv(dtype=DType.bf16, length=16)]),
        ]

    def block_indices(self, src_metas, dst_metas):
        return [()]

    def validate(self, src_metas, dst_metas):
        x_meta = src_metas[0]
        self._n = x_meta.shape[-1]
