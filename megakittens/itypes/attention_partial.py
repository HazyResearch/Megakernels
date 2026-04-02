"""
Single-partition decode attention instruction type.
Computes: softmax(Q @ K^T * scale) @ V per KV head.
Uses pipelined TMA loading of KV cache tiles with online softmax.
"""

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorSpec
from ..jit.pykittens import sv, st


class AttentionPartial(IType):

    def __init__(self, head_dim: int = 64, kv_block_size: int = 16, gqa_ratio: int = 4):
        self._head_dim = head_dim
        self._kv_block_size = kv_block_size
        self._gqa_ratio = gqa_ratio

    @property
    def name(self):
        return "attention_partial"

    @property
    def cpp_template(self):
        return (f"AttentionPartial<MKConfig, MKGlobals, "
                f"{self._head_dim}, {self._kv_block_size}, {self._gqa_ratio}, {{tensors}}>")

    @property
    def cpp_include(self):
        return "itypes/llama1b/attention_partial.cuh"

    @property
    def op_type(self):
        return "attention_partial"

    @property
    def inputs(self):
        return [
            # SRC_Q: q_post_rope — loaded via cp.async (raw_ptr) in consumer
            TensorSpec(dtype=DType.bf16, granularity=(1,)),
            # SRC_K_CACHE: k_cache — TMA tile loads (st<kv_block, head_dim>, axis=1 for depth)
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1),
                       tma_types=[
                           sv(dtype=DType.bf16, length=self._kv_block_size),
                           st(dtype=DType.bf16, rows=self._kv_block_size,
                              cols=self._head_dim, axis=1),
                       ]),
            # SRC_V_CACHE: v_cache — same TMA types as k_cache
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1),
                       tma_types=[
                           sv(dtype=DType.bf16, length=self._kv_block_size),
                           st(dtype=DType.bf16, rows=self._kv_block_size,
                              cols=self._head_dim, axis=1),
                       ]),
        ]

    @property
    def outputs(self):
        return [
            # DST: attn_out — TMA sv store (head_dim-element chunks)
            TensorSpec(dtype=DType.bf16, granularity=(1,),
                       tma_types=[sv(dtype=DType.bf16, length=self._head_dim)]),
        ]

    def block_indices(self, src_metas, dst_metas):
        return [()]

    def validate(self, src_metas, dst_metas):
        pass
