import torch
import torch.nn.functional as F

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorSpec
from ...jit.pykittens import sv, st


@torch.library.custom_op("megakittens::attention_partial", mutates_args=())
def attention_partial_op(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    pos_id: torch.Tensor, attn_scale: torch.Tensor,
) -> torch.Tensor:
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    gqa_ratio = q.shape[0] // (num_kv_heads * head_dim)
    seq_len = pos_id.item() + 1
    scale = attn_scale.item()
    qh = q.view(num_kv_heads, gqa_ratio, head_dim)
    k = k_cache[0, :seq_len].permute(1, 2, 0)
    v = k_cache[0, :seq_len].permute(1, 0, 2)
    scores = torch.bmm(qh, k) * scale
    w = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.bmm(w, v).reshape(-1)


@attention_partial_op.register_fake
def _attention_partial_fake(q, k_cache, v_cache, pos_id, attn_scale):
    return torch.empty_like(q)


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
            TensorSpec(dtype=DType.int32, granularity=(1,)),                             # pos_id
            TensorSpec(dtype=DType.fp32, granularity=(1,)),                              # attn_scale
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,),                               # attn_out
                       tma_types=[sv(dtype=DType.bf16, length=self._head_dim)]),
        ]

    def block_indices(self, src_metas, dst_metas):
        return [()]

    def test_args(self, case):
        return ()

    def access_regions(self, block_index, src_metas, dst_metas):
        return [], []

    def validate(self, src_metas, dst_metas):
        pass
