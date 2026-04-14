import torch

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorSpec
from ...jit.pykittens import sv, st


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
def _rms_lm_head_fake(x, norm_weight, lm_head, eps):
    return torch.empty((x.shape[0], lm_head.shape[0]), dtype=x.dtype, device=x.device)


def _resolve_rms_lm_head(args, kwargs):
    x = args[0]
    return RmsLmHead(n=x.shape[-1])


class RmsLmHead(IType):

    torch_functions_map = {
        torch.ops.megakittens.rms_lm_head: _resolve_rms_lm_head,
        torch.ops.megakittens.rms_lm_head.default: _resolve_rms_lm_head,
    }

    def __init__(self, n):
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
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,),                           # hidden_states
                       tma_types=[sv(dtype=DType.bf16, length=self._n)]),
            TensorSpec(dtype=DType.bf16, granularity=(1,),                           # norm_weight
                       tma_types=[sv(dtype=DType.bf16, length=self._n)]),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1),                         # lm_head_weights
                       tma_types=[st(dtype=DType.bf16, rows=16, cols=512)]),
            TensorSpec(dtype=DType.fp32, granularity=(1,)),                         # rms_norm_eps
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,),                               # logits
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
