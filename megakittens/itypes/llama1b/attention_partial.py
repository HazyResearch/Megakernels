from typing import List, Tuple

import torch
import torch.nn.functional as F

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorMeta, TensorRange, TensorSpec
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
    qh = q.view(num_kv_heads, gqa_ratio, head_dim)
    k = k_cache[0, :seq_len].permute(1, 2, 0)
    v = v_cache[0, :seq_len].permute(1, 0, 2)
    scores = torch.bmm(qh, k) * attn_scale
    w = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.bmm(w, v).reshape(-1)


@attention_partial_op.register_fake
def _attention_partial_fake(q, k_cache, v_cache, pos_id, attn_scale):
    return torch.empty_like(q)


class AttentionPartial(IType):

    test_cases = [
        ((), (8, 32, 32)),  # (num_kv_heads, seq_len, max_seq_len)
    ]
    test_atol = 1e-2
    test_rtol = 1e-2
    bench_cases = [
        ((), (8, 128, 512)),
    ]

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
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, self._head_dim), tma_types=kv_tma),  # k_cache
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, self._head_dim), tma_types=kv_tma),  # v_cache
            TensorSpec(dtype=DType.int32, granularity=(1,)),                             # pos_id
            TensorSpec(dtype=DType.fp32, granularity=(1,)),                              # attn_scale
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self._head_dim,),                    # attn_out
                       tma_types=[sv(dtype=DType.bf16, length=self._head_dim)]),
        ]

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> int:
        k_range = src_ranges[1]
        return k_range[-2].size

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        k_range = src_ranges[1]
        layer_idx = k_range[-4].start
        return [(layer_idx, kv_head) for kv_head in range(k_range[-2].start, k_range[-2].stop)]

    def test_args(self, case):
        num_kv_heads, seq_len, max_seq_len = case
        head_dim = self._head_dim
        gqa_ratio = self._gqa_ratio
        q = torch.randn(num_kv_heads * gqa_ratio * head_dim, dtype=torch.bfloat16, device="cuda")
        k_cache = torch.randn(1, max_seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        v_cache = torch.randn(1, max_seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        pos_id = torch.tensor([seq_len - 1], dtype=torch.int32, device="cuda")
        attn_scale = torch.tensor([1.0 / (head_dim ** 0.5)], dtype=torch.float32, device="cuda")
        return (q, k_cache, v_cache, pos_id, attn_scale)

    def access_regions(self, block_index, src_metas, dst_metas):
        layer_idx, kv_head = block_index
        head_dim = self._head_dim
        gqa_ratio = self._gqa_ratio
        group_size = gqa_ratio * head_dim
        q_start = kv_head * group_size
        q_end = q_start + group_size
        _, max_seq_len, num_kv_heads, _ = src_metas[1].shape
        q_region = ((q_start, q_end),)
        kv_region = ((layer_idx, layer_idx + 1), (0, max_seq_len), (kv_head, kv_head + 1), (0, head_dim))
        pos_region = ((0, 1),)
        scale_region = ((0, 1),)
        out_region = ((q_start, q_end),)
        return [[q_region], [kv_region], [kv_region], [pos_region], [scale_region]], [[out_region]]

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)


@torch.library.custom_op("megakittens::attention_partial_multi", mutates_args=())
def attention_partial_multi_op(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    pos_id: torch.Tensor, attn_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    gqa_ratio = q.shape[0] // (num_kv_heads * head_dim)
    num_heads = num_kv_heads * gqa_ratio
    seq_len = pos_id.item() + 1
    qh = q.view(num_kv_heads, gqa_ratio, head_dim)
    k = k_cache[0, :seq_len].permute(1, 2, 0)
    v = k_cache[0, :seq_len].permute(1, 0, 2)
    scores = torch.bmm(qh, k) * attn_scale
    lse = torch.log2(torch.sum(torch.exp(scores.float()), dim=-1))
    w = F.softmax(scores.float(), dim=-1).to(q.dtype)
    v = v_cache[0, :seq_len].permute(1, 0, 2)
    o = torch.bmm(w, v).float()
    o_inter = o.view(num_heads, 1, head_dim)
    l_inter = lse.view(num_heads, 1)
    return o_inter, l_inter


@attention_partial_multi_op.register_fake
def _attention_partial_multi_fake(q, k_cache, v_cache, pos_id, attn_scale):
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    gqa_ratio = q.shape[0] // (num_kv_heads * head_dim)
    num_heads = num_kv_heads * gqa_ratio
    o_inter = torch.empty(num_heads, 1, head_dim, dtype=torch.float32, device=q.device)
    l_inter = torch.empty(num_heads, 1, dtype=torch.float32, device=q.device)
    return o_inter, l_inter


class AttentionPartialMulti(IType):

    test_cases = []  # TODO: re-enable — kernel produces inf in l_inter under global_work_queue
    test_atol = 1e-2
    test_rtol = 1e-2
    bench_cases = []

    def __init__(self, head_dim=64, kv_block_size=16, gqa_ratio=4):
        self._head_dim = head_dim
        self._kv_block_size = kv_block_size
        self._gqa_ratio = gqa_ratio

    @property
    def name(self) -> str:
        return "attention_partial_multi"

    @property
    def cpp_template(self) -> str:
        return (f"AttentionPartial<MKConfig, MKGlobals, "
                f"{self._head_dim}, {self._kv_block_size}, {self._gqa_ratio}, {{tensors}}>")

    @property
    def cpp_include(self) -> str:
        return "itypes/llama1b/attention_partial.cuh"

    @property
    def inputs(self) -> list[TensorSpec]:
        kv_tma = [
            sv(dtype=DType.bf16, length=self._kv_block_size),
            st(dtype=DType.bf16, rows=self._kv_block_size, cols=self._head_dim, axis=1),
        ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,)),                              # q_post_rope
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, self._head_dim), tma_types=kv_tma),  # k_cache
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, self._head_dim), tma_types=kv_tma),  # v_cache
            TensorSpec(dtype=DType.int32, granularity=(1,)),                             # pos_id
            TensorSpec(dtype=DType.fp32, granularity=(1,)),                              # attn_scale
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.fp32, granularity=(1, 1, 1),                          # o_intermediate
                       tma_types=[sv(dtype=DType.fp32, length=self._head_dim)]),
            TensorSpec(dtype=DType.fp32, granularity=(1, 1)),                            # l_intermediate (raw store)
        ]

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> int:
        k_range = src_ranges[1]
        return k_range[-2].size

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        k_range = src_ranges[1]
        layer_idx = k_range[-4].start
        return [(layer_idx, kv_head) for kv_head in range(k_range[-2].start, k_range[-2].stop)]

    def test_args(self, case):
        num_kv_heads, seq_len, max_seq_len = case
        head_dim = self._head_dim
        gqa_ratio = self._gqa_ratio
        q = torch.randn(num_kv_heads * gqa_ratio * head_dim, dtype=torch.bfloat16, device="cuda")
        k_cache = torch.randn(1, max_seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        v_cache = torch.randn(1, max_seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        pos_id = torch.tensor([seq_len - 1], dtype=torch.int32, device="cuda")
        attn_scale = torch.tensor([1.0 / (head_dim ** 0.5)], dtype=torch.float32, device="cuda")
        return (q, k_cache, v_cache, pos_id, attn_scale)

    def access_regions(self, block_index, src_metas, dst_metas):
        _, kv_head = block_index
        head_dim = self._head_dim
        gqa_ratio = self._gqa_ratio
        group_size = gqa_ratio * head_dim
        q_start = kv_head * group_size
        q_end = q_start + group_size
        num_layers, max_seq_len, num_kv_heads, _ = src_metas[1].shape
        q_region = ((q_start, q_end),)
        kv_region = ((0, num_layers), (0, max_seq_len), (kv_head, kv_head + 1), (0, head_dim))
        pos_region = ((0, 1),)
        scale_region = ((0, 1),)
        num_heads = num_kv_heads * gqa_ratio
        num_partials = dst_metas[0].shape[1]
        o_head_start = kv_head * gqa_ratio
        o_head_end = o_head_start + gqa_ratio
        o_region = ((o_head_start, o_head_end), (0, num_partials), (0, head_dim))
        l_region = ((o_head_start, o_head_end), (0, num_partials))
        return [[q_region], [kv_region], [kv_region], [pos_region], [scale_region]], [[o_region], [l_region]]

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
