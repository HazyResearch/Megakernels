import torch

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorSpec
from ...jit.pykittens import sv, st


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


def _resolve_rms_upgate_silu(args, kwargs):
    x = args[0]
    return RmsUpgateSilu(n=x.shape[-1])


class RmsUpgateSilu(IType):

    torch_functions_map = {
        torch.ops.megakittens.rms_upgate_silu: _resolve_rms_upgate_silu,
        torch.ops.megakittens.rms_upgate_silu.default: _resolve_rms_upgate_silu,
    }

    def __init__(self, n):
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
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,),                           # hidden_states
                       tma_types=[sv(dtype=DType.bf16, length=self._n)]),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1),                         # norm_weight
                       tma_types=[sv(dtype=DType.bf16, length=self._n)]),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1),                      # up_weights
                       tma_types=[st(dtype=DType.bf16, rows=16, cols=512)]),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1),                      # gate_weights
                       tma_types=[st(dtype=DType.bf16, rows=16, cols=512)]),
            TensorSpec(dtype=DType.fp32, granularity=(1,)),                         # rms_norm_eps
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,),                               # silu_out
                       tma_types=[sv(dtype=DType.bf16, length=16)]),
        ]

    def block_indices(self, src_metas, dst_metas):
        return [()]

    def test_args(self, case):
        return ()

    def access_regions(self, block_index, src_metas, dst_metas):
        return [], []

    def validate(self, src_metas, dst_metas):
        super().validate(src_metas, dst_metas)
        if src_metas[0].shape[-1] != self._n:
            raise RuntimeError(
                f"[MegaKittens] {self.name}: expected n={self._n}, got {src_metas[0].shape[-1]}"
            )
