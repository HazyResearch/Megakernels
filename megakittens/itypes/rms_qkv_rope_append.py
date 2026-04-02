"""
RMSNorm + QKV matvec + RoPE + KV cache write instruction type.
First instruction in each transformer layer during decode.
"""

from typing import List, Tuple

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorSpec
from ..jit.pykittens import sv, st


class RmsQkvRopeAppend(IType):

    def __init__(self, n: int = 0, head_dim: int = 64, num_kv_heads: int = 8) -> None:
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
                # SRC_ACT: hidden_states — TMA sv load into activation page
                TensorSpec(dtype=DType.bf16, granularity=(1,),
                           tma_types=[sv(dtype=DType.bf16, length=self._n)]),
                # SRC_NORM: attn_norm_weights — TMA sv load into activation page
                TensorSpec(dtype=DType.bf16, granularity=(1, 1),
                           tma_types=[sv(dtype=DType.bf16, length=self._n)]),
                # SRC_QKV_W: qkv_weights — TMA st load (16×512 tiles)
                TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1),
                           tma_types=[st(dtype=DType.bf16, rows=16, cols=512)]),
                # SRC_ROPE_COS: rope_cos — loaded via raw_ptr, not TMA.
                # Dummy sv TMA type ensures alignment consistency in Globals struct.
                TensorSpec(dtype=DType.fp32, granularity=(1, 1),
                           tma_types=[sv(dtype=DType.fp32, length=self._head_dim)]),
                # SRC_ROPE_SIN: rope_sin — same as above
                TensorSpec(dtype=DType.fp32, granularity=(1, 1),
                           tma_types=[sv(dtype=DType.fp32, length=self._head_dim)]),
                # SRC_K_CACHE: k_cache — TMA sv store (16-element chunks)
                TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1),
                           tma_types=[sv(dtype=DType.bf16, length=16)]),
                # SRC_V_CACHE: v_cache — TMA sv store (16-element chunks)
                TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1),
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
            # DST_Q: q_post_rope — TMA sv store (16-element chunks)
            TensorSpec(dtype=DType.bf16, granularity=(1,),
                       tma_types=[sv(dtype=DType.bf16, length=16)]),
        ]

    def block_indices(
        self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...],
    ) -> List[Tuple[int, ...]]:
        return [()]

    def validate(
        self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...],
    ) -> None:
        self._n = src_metas[0].shape[0]
