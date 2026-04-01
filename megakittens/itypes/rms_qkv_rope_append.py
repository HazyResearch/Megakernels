"""
RMSNorm + QKV matvec + RoPE + KV cache write instruction type.
First instruction in each transformer layer during decode.
"""

from typing import List, Tuple

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorSpec


class RmsQkvRopeAppend(IType):

    def __init__(self, n: int = 0) -> None:
        self._n = n

    @property
    def name(self) -> str:
        return "rms_qkv_rope_append"

    @property
    def cpp_template(self) -> str:
        return f"RmsQkvRopeAppend<MKConfig, MKGlobals, {self._n}, {{tensors}}>"

    @property
    def cpp_include(self) -> str:
        return "itypes/rms_qkv_rope_append.cuh"

    @property
    def op_type(self) -> str:
        return "rms_qkv_rope_append"

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,)),           # hidden_states
            TensorSpec(dtype=DType.bf16, granularity=(1, 1)),         # attn_norm_weights
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1)),     # qkv_weights
            TensorSpec(dtype=DType.fp32, granularity=(1, 1)),         # rope_cos
            TensorSpec(dtype=DType.fp32, granularity=(1, 1)),         # rope_sin
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1)),  # k_cache
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1)),  # v_cache
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [TensorSpec(dtype=DType.bf16, granularity=(1,))]  # q_post_rope

    def block_indices(
        self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...],
    ) -> List[Tuple[int, ...]]:
        return [()]

    def validate(
        self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...],
    ) -> None:
        self._n = src_metas[0].shape[0]
