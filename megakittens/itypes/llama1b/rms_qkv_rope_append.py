from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorSpec
from ...jit.pykittens import sv, st


class RmsQkvRopeAppend(IType):

    def __init__(self, n=0, head_dim=64, num_kv_heads=8):
        self._n = n
        self._head_dim = head_dim
        self._num_kv_heads = num_kv_heads

    @property
    def name(self) -> str:
        return "rms_qkv_rope_append"

    @property
    def cpp_template(self) -> str:
        return (f"RmsQkvRopeAppend<MKConfig, MKGlobals, {self._n}, "
                f"{self._head_dim}, {self._num_kv_heads}, {{tensors}}>")

    @property
    def cpp_include(self) -> str:
        return "itypes/llama1b/rms_qkv_rope_append.cuh"

    @property
    def op_type(self) -> str:
        return "rms_qkv_rope_append"

    @property
    def inputs(self) -> list[TensorSpec]:
        if self._n > 0:
            return [
                TensorSpec(dtype=DType.bf16, granularity=(1,),                           # hidden_states
                           tma_types=[sv(dtype=DType.bf16, length=self._n)]),
                TensorSpec(dtype=DType.bf16, granularity=(1, 1),                         # attn_norm_weights
                           tma_types=[sv(dtype=DType.bf16, length=self._n)]),
                TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1),                      # qkv_weights
                           tma_types=[st(dtype=DType.bf16, rows=16, cols=512)]),
                TensorSpec(dtype=DType.fp32, granularity=(1, 1),                         # rope_cos
                           tma_types=[sv(dtype=DType.fp32, length=self._head_dim)]),
                TensorSpec(dtype=DType.fp32, granularity=(1, 1),                         # rope_sin
                           tma_types=[sv(dtype=DType.fp32, length=self._head_dim)]),
                TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1),                   # k_cache
                           tma_types=[sv(dtype=DType.bf16, length=16)]),
                TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1),                   # v_cache
                           tma_types=[sv(dtype=DType.bf16, length=16)]),
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1)),
            TensorSpec(dtype=DType.fp32, granularity=(1, 1)),
            TensorSpec(dtype=DType.fp32, granularity=(1, 1)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1)),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,),                               # q_post_rope
                       tma_types=[sv(dtype=DType.bf16, length=16)]),
        ]

    def block_indices(self, src_metas, dst_metas):
        return [()]

    def validate(self, src_metas, dst_metas):
        self._n = src_metas[0].shape[0]
