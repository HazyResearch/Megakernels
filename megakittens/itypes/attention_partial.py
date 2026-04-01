"""
Single-partition decode attention instruction type.
Computes: softmax(Q @ K^T * scale) @ V per KV head.
"""

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorSpec


class AttentionPartial(IType):

    @property
    def name(self):
        return "attention_partial"

    @property
    def cpp_template(self):
        return "AttentionPartial<MKConfig, MKGlobals, {tensors}>"

    @property
    def cpp_include(self):
        return "itypes/attention_partial.cuh"

    @property
    def op_type(self):
        return "attention_partial"

    @property
    def inputs(self):
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,)),          # q_post_rope
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1)),  # k_cache
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1)),  # v_cache
        ]

    @property
    def outputs(self):
        return [TensorSpec(dtype=DType.bf16, granularity=(1,))]  # attn_out

    def block_indices(self, src_metas, dst_metas):
        return [()]

    def validate(self, src_metas, dst_metas):
        pass
