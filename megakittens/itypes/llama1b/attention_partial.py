from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorSpec
from ...jit.pykittens import sv, st


class AttentionPartial(IType):

    def __init__(self, head_dim=64, kv_block_size=16, gqa_ratio=4):
        self._head_dim = head_dim
        self._kv_block_size = kv_block_size
        self._gqa_ratio = gqa_ratio

    @property
    def name(self) -> str:
        return "attention_partial"

    @property
    def cpp_template(self) -> str:
        return (f"AttentionPartial<MKConfig, MKGlobals, "
                f"{self._head_dim}, {self._kv_block_size}, {self._gqa_ratio}, {{tensors}}>")

    @property
    def cpp_include(self) -> str:
        return "itypes/llama1b/attention_partial.cuh"

    @property
    def op_type(self) -> str:
        return "attention_partial"

    @property
    def inputs(self) -> list[TensorSpec]:
        kv_tma = [
            sv(dtype=DType.bf16, length=self._kv_block_size),
            st(dtype=DType.bf16, rows=self._kv_block_size, cols=self._head_dim, axis=1),
        ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,)),                              # q_post_rope
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1), tma_types=kv_tma),    # k_cache
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1), tma_types=kv_tma),    # v_cache
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,),                               # attn_out
                       tma_types=[sv(dtype=DType.bf16, length=self._head_dim)]),
        ]

    def block_indices(self, src_metas, dst_metas):
        return [()]

    def validate(self, src_metas, dst_metas):
        pass
