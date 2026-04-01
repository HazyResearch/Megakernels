"""
Fused RMSNorm + up/gate dual matvec + SiLU gating instruction type for decode.
Computes: silu(gate_weights @ rms_norm(x)) * (up_weights @ rms_norm(x))
"""

import torch

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorSpec
from ..jit.pykittens import sv


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

    def __init__(self, n: int = 0):
        self._n = n

    @property
    def name(self):
        return "rms_upgate_silu"

    @property
    def cpp_template(self):
        return f"RmsUpgateSilu<MKConfig, MKGlobals, {self._n}, {{tensors}}>"

    @property
    def cpp_include(self):
        return "itypes/rms_upgate_silu.cuh"

    @property
    def op_type(self):
        return "rms_upgate_silu"

    @property
    def inputs(self):
        if self._n > 0:
            return [
                TensorSpec(dtype=DType.bf16, granularity=(1,),
                           tma_types=[sv(dtype=DType.bf16, length=self._n)]),
                TensorSpec(dtype=DType.bf16, granularity=(1, 1)),
                TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1)),
                TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1)),
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1)),
        ]

    @property
    def outputs(self):
        return [TensorSpec(dtype=DType.bf16, granularity=(1,))]

    def block_indices(self, src_metas, dst_metas):
        return [()]

    def validate(self, src_metas, dst_metas):
        x_meta = src_metas[0]
        self._n = x_meta.shape[-1]
