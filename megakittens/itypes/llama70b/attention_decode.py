from typing import List, Tuple

import torch

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorMeta, TensorRange, TensorSpec
from ...jit.pykittens import st, sv


HIDDEN_DIM = 8192
HEAD_DIM = 128
NUM_Q_HEADS = 64
NUM_KV_HEADS = 8
GQA_RATIO = NUM_Q_HEADS // NUM_KV_HEADS
PAGE_SIZE = 128
KV_BLOCK_SIZE = 16


@torch.library.custom_op("megakittens::attention_decode70b", mutates_args=())
def attention_decode70b_op(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos_id: torch.Tensor,
    attn_scale: torch.Tensor,
) -> torch.Tensor:
    B = q.shape[-2]
    num_pages = k_cache.shape[-4]
    pages_per_seq = num_pages // B
    seq_len = pos_id.item() + 1
    num_pages_to_read = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE

    q_heads = q.view(B, NUM_Q_HEADS, HEAD_DIM).float()
    out = torch.empty_like(q_heads)
    scale = attn_scale.item()

    for seq_idx in range(B):
        page_start = seq_idx * pages_per_seq
        k_hist = k_cache[page_start:page_start + num_pages_to_read].reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:seq_len]
        v_hist = v_cache[page_start:page_start + num_pages_to_read].reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:seq_len]
        q_grouped = q_heads[seq_idx].view(NUM_KV_HEADS, GQA_RATIO, HEAD_DIM)
        scores = torch.einsum("kgd,tkd->kgt", q_grouped, k_hist.float()) * scale
        weights = torch.softmax(scores, dim=-1)
        seq_out = torch.einsum("kgt,tkd->kgd", weights, v_hist.float())
        out[seq_idx] = seq_out.reshape(NUM_Q_HEADS, HEAD_DIM).to(out.dtype)

    return out.reshape(B, HIDDEN_DIM).to(q.dtype)


@attention_decode70b_op.register_fake
def _attention_decode70b_fake(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos_id: torch.Tensor,
    attn_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(q)


def _resolve_attention_decode70b(args, kwargs):
    q_shape = args[0].meta["val"].shape
    k_shape = args[1].meta["val"].shape
    return AttentionDecode70b(batch_size=q_shape[-2], num_pages=k_shape[-4])


class AttentionDecode70b(IType):
    Q_TMA = sv(dtype=DType.bf16, length=HIDDEN_DIM)
    KV_TMA = st(dtype=DType.bf16, rows=KV_BLOCK_SIZE, cols=HEAD_DIM, axis=1)
    O_TMA = sv(dtype=DType.bf16, length=HEAD_DIM)

    torch_functions_map = {
        torch.ops.megakittens.attention_decode70b: _resolve_attention_decode70b,
        torch.ops.megakittens.attention_decode70b.default: _resolve_attention_decode70b,
    }

    test_cases = [
        ((256, 256), (256, 128, 15)),
        ((128, 128), (128, 128, 62)),
        ((128, 128), (128, 128, 63)),
        ((256, 256), (256, 256, 127)),
        ((512, 512), (512, 512, 255)),
        ((1024, 1024), (1024, 128, 62)),
        ((1024, 1024), (1024, 128, 127)),
    ]
    test_atol = 1e-2
    test_rtol = 1e-2
    bench_cases = [
        ((1024, 1024), (1024, 1024, 511)),
    ]

    def __init__(self, batch_size: int = 0, num_pages: int = 0):
        self.batch_size = batch_size
        self.num_pages = num_pages

    @property
    def pages_per_seq(self) -> int:
        return self.num_pages // self.batch_size if self.batch_size else 0

    @property
    def cpp_template(self) -> str:
        return (
            f"llama70b::AttentionDecode<MKConfig, MKGlobals, "
            f"{self.batch_size}, {self.num_pages}, {self.pages_per_seq}, "
            f"{HEAD_DIM}, {PAGE_SIZE}, {KV_BLOCK_SIZE}, {NUM_Q_HEADS}, {NUM_KV_HEADS}, "
            f"{{tensors}}>"
        )

    @property
    def cpp_include(self) -> str:
        return "itypes/llama70b/attention_decode.cuh"

    def test_args(self, case: tuple) -> tuple:
        B, max_seq_len, pos = case
        pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        num_pages = B * pages_per_seq
        return (
            torch.randn(B, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"),
            torch.randn(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"),
            torch.randn(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"),
            torch.tensor([pos], dtype=torch.int32, device="cuda"),
            torch.tensor([1.0 / (HEAD_DIM ** 0.5)], dtype=torch.float32, device="cuda"),
        )

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1, HEAD_DIM), tma_types=[self.Q_TMA]),
            TensorSpec(dtype=DType.bf16, granularity=(1, KV_BLOCK_SIZE, 1, HEAD_DIM), tma_types=[self.KV_TMA]),
            TensorSpec(dtype=DType.bf16, granularity=(1, KV_BLOCK_SIZE, 1, HEAD_DIM), tma_types=[self.KV_TMA]),
            TensorSpec(dtype=DType.int32, granularity=(1,)),
            TensorSpec(dtype=DType.fp32, granularity=(1,)),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1, HEAD_DIM), tma_types=[self.O_TMA]),
        ]

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> int:
        return dst_ranges[0][-2].size

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        k_range = src_ranges[1]
        out_range = dst_ranges[0]
        base_page = k_range[-4].start
        return [
            (base_page, out_range[-2].start + seq_idx)
            for seq_idx in range(out_range[-2].size)
        ]

    def access_regions(
        self,
        block_index: Tuple[int, ...],
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ):
        base_page, seq_idx = block_index
        pages_per_seq = self.num_pages // self.batch_size
        page_start = base_page + seq_idx * pages_per_seq
        page_stop = page_start + pages_per_seq

        q_region = ((seq_idx, seq_idx + 1), (0, HIDDEN_DIM))
        k_region = ((page_start, page_stop), (0, PAGE_SIZE), (0, NUM_KV_HEADS), (0, HEAD_DIM))
        v_region = k_region
        pos_region = ((0, 1),)
        scale_region = ((0, 1),)
        out_region = q_region
        return (
            [[q_region], [k_region], [v_region], [pos_region], [scale_region]],
            [[out_region]],
        )

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)

        q_shape = src_ranges[0].effective_shape
        k_shape = src_ranges[1].effective_shape
        v_shape = src_ranges[2].effective_shape
        pos_shape = src_ranges[3].effective_shape
        scale_shape = src_ranges[4].effective_shape
        out_shape = dst_ranges[0].effective_shape

        B = q_shape[-2]
        num_pages = k_shape[-4]

        if q_shape[-1] != HIDDEN_DIM:
            raise RuntimeError(
                f"[MegaKittens] AttentionDecode70b expected q dim={HIDDEN_DIM}, got {q_shape[-1]}"
            )
        if out_shape[-2:] != q_shape[-2:]:
            raise RuntimeError(
                f"[MegaKittens] AttentionDecode70b output shape mismatch: q={q_shape[-2:]} out={out_shape[-2:]}"
            )
        if k_shape != v_shape:
            raise RuntimeError(
                f"[MegaKittens] AttentionDecode70b K/V cache shape mismatch: k={k_shape}, v={v_shape}"
            )
        if k_shape[-3:] != (PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM):
            raise RuntimeError(
                f"[MegaKittens] AttentionDecode70b expected cache trailing shape "
                f"({PAGE_SIZE}, {NUM_KV_HEADS}, {HEAD_DIM}), got {k_shape[-3:]}"
            )
        if num_pages % B != 0:
            raise RuntimeError(
                f"[MegaKittens] AttentionDecode70b requires num_pages divisible by B, got num_pages={num_pages}, B={B}"
            )
        if pos_shape[-1] != 1:
            raise RuntimeError(
                f"[MegaKittens] AttentionDecode70b expected pos_id shape [1], got effective shape {pos_shape}"
            )
        if scale_shape[-1] != 1:
            raise RuntimeError(
                f"[MegaKittens] AttentionDecode70b expected attn_scale shape [1], got effective shape {scale_shape}"
            )

        if B != self.batch_size:
            raise RuntimeError(
                f"[MegaKittens] AttentionDecode70b expected B={self.batch_size}, got {B}"
            )
        if num_pages != self.num_pages:
            raise RuntimeError(
                f"[MegaKittens] AttentionDecode70b expected num_pages={self.num_pages}, got {num_pages}"
            )
