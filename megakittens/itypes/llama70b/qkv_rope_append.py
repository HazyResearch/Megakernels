from typing import List, Tuple

import torch

from ...jit.pykittens import st, sv
from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorMeta, TensorRange, TensorSpec


HIDDEN_DIM = 8192
HEAD_DIM = 128
NUM_Q_HEADS = 64
NUM_KV_HEADS = 8
Q_DIM = NUM_Q_HEADS * HEAD_DIM
KV_DIM = NUM_KV_HEADS * HEAD_DIM
QKV_DIM = Q_DIM + 2 * KV_DIM
PAGE_SIZE = 128

Mb = 256
Nb = 256
Kb = 64
EPI_PIPE_DEPTH = 8
NUM_CONSUMERS = 2
M_INST = NUM_CONSUMERS * Mb


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x, cos, and sin are pre-interleaved as [first_half_0, second_half_0, ...].
    x_float = x.float()
    x_even = x_float[..., ::2]
    x_odd = x_float[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    return (x_float * cos + rotated * sin).to(x.dtype)


def _interleave_indices(num_heads: int, head_dim: int, device=None) -> torch.Tensor:
    half = head_dim // 2
    indices = []
    for head in range(num_heads):
        offset = head * head_dim
        for i in range(half):
            indices.append(offset + i)
            indices.append(offset + half + i)
    return torch.tensor(indices, dtype=torch.long, device=device)


def interleave_qkv_weights(qkv_weights: torch.Tensor) -> torch.Tensor:
    """Interleave Q/K rows so CUDA can apply RoPE with lane-neighbor shuffles."""
    q_indices = _interleave_indices(NUM_Q_HEADS, HEAD_DIM, device=qkv_weights.device)
    k_indices = _interleave_indices(NUM_KV_HEADS, HEAD_DIM, device=qkv_weights.device)
    q = qkv_weights[..., :Q_DIM, :][..., q_indices, :]
    k = qkv_weights[..., Q_DIM:Q_DIM + KV_DIM, :][..., k_indices, :]
    v = qkv_weights[..., Q_DIM + KV_DIM:, :]
    return torch.cat((q, k, v), dim=-2)


def interleave_rope_table(table: torch.Tensor) -> torch.Tensor:
    indices = _interleave_indices(1, HEAD_DIM, device=table.device)
    return table[..., indices]


@torch.library.custom_op("megakittens::qkv_rope_append70b", mutates_args=("k_cache", "v_cache"))
def qkv_rope_append70b_op(
    x: torch.Tensor,
    qkv_weights: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    pos_ids: torch.Tensor,
    append_ids: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    B = x.shape[-2]

    qkv = x.float() @ qkv_weights[0].float().transpose(-1, -2)
    q = qkv[..., :Q_DIM].view(B, NUM_Q_HEADS, HEAD_DIM).to(x.dtype)
    k = qkv[..., Q_DIM:Q_DIM + KV_DIM].view(B, NUM_KV_HEADS, HEAD_DIM).to(x.dtype)
    v = qkv[..., Q_DIM + KV_DIM:].view(B, NUM_KV_HEADS, HEAD_DIM).to(x.dtype)

    cos = rope_cos[pos_ids.long()].view(B, 1, HEAD_DIM)
    sin = rope_sin[pos_ids.long()].view(B, 1, HEAD_DIM)
    q = _apply_rope(q, cos, sin)
    k = _apply_rope(k, cos, sin)

    for seq_idx in range(B):
        # Assumes synchronous decode and a static paged KV cache where sequence i
        # owns pages [i * pages_per_seq, (i + 1) * pages_per_seq). append_ids
        # are pre-flattened in the layer-local page space as page * PAGE_SIZE + offset.
        append_idx = int(append_ids[seq_idx].item())
        page_idx = append_idx // PAGE_SIZE
        page_offset = append_idx % PAGE_SIZE
        k_cache[page_idx, page_offset] = k[seq_idx]
        v_cache[page_idx, page_offset] = v[seq_idx]

    return q.reshape(B, HIDDEN_DIM).to(x.dtype)


@qkv_rope_append70b_op.register_fake
def _qkv_rope_append70b_fake(
    x: torch.Tensor,
    qkv_weights: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    pos_ids: torch.Tensor,
    append_ids: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


def _resolve_qkv_rope_append70b(args, kwargs):
    x_shape = args[0].meta["val"].shape
    w_shape = args[1].meta["val"].shape
    k_shape = args[6].meta["val"].shape
    num_pages_per_layer = k_shape[-4] // w_shape[-3]
    return QkvRopeAppend70b(batch_size=x_shape[-2], num_pages=num_pages_per_layer), [0, 1, 2]


class QkvRopeAppend70b(IType):
    Mb = Mb
    Nb = Nb
    Kb = Kb
    EPI_PIPE_DEPTH = EPI_PIPE_DEPTH
    NUM_CONSUMERS = NUM_CONSUMERS
    M_INST = M_INST

    X_TMA = st(dtype=DType.bf16, rows=Mb // 2, cols=Kb)
    W_TMA = st(dtype=DType.bf16, rows=Nb // 2, cols=Kb)
    Q_TMA = st(dtype=DType.bf16, rows=Mb // 2, cols=Nb // EPI_PIPE_DEPTH)
    ROPE_TMA = sv(dtype=DType.fp32, length=HEAD_DIM)
    KV_TMA = sv(dtype=DType.bf16, length=HEAD_DIM)

    torch_functions_map = {
        torch.ops.megakittens.qkv_rope_append70b: _resolve_qkv_rope_append70b,
        torch.ops.megakittens.qkv_rope_append70b.default: _resolve_qkv_rope_append70b,
    }

    test_cases = [
        ((512, 512), (512, 128, 62)),
        ((1024, 2048), (1024, 256, 127)),
    ]
    test_atol = 1e-1
    test_rtol = 1e-2
    bench_cases = [
        ((1024, 2048), (1024, 256, 127)),
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
            f"llama70b::QkvRopeAppend<MKConfig, MKGlobals, "
            f"{self.batch_size}, {self.num_pages}, {self.pages_per_seq}, "
            f"{HIDDEN_DIM}, {QKV_DIM}, {HEAD_DIM}, {PAGE_SIZE}, "
            f"{NUM_Q_HEADS}, {NUM_KV_HEADS}, {{tensors}}>"
        )

    @property
    def cpp_include(self) -> str:
        return "itypes/llama70b/qkv_rope_append.cuh"

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.M_INST, self.Kb), tma_types=[self.X_TMA]),
            TensorSpec(dtype=DType.bf16, granularity=(1, self.Nb, self.Kb), tma_types=[self.W_TMA]),
            TensorSpec(dtype=DType.fp32, granularity=(1, HEAD_DIM), tma_types=[self.ROPE_TMA]),
            TensorSpec(dtype=DType.fp32, granularity=(1, HEAD_DIM), tma_types=[self.ROPE_TMA]),
            TensorSpec(dtype=DType.int32, granularity=(self.M_INST,)),
            TensorSpec(dtype=DType.int32, granularity=(self.M_INST,)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, HEAD_DIM), tma_types=[self.KV_TMA]),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, HEAD_DIM), tma_types=[self.KV_TMA]),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.M_INST, self.Nb), tma_types=[self.Q_TMA]),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, HEAD_DIM), tma_types=[self.KV_TMA]),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, HEAD_DIM), tma_types=[self.KV_TMA]),
        ]

    @property
    def inplace_mapping(self) -> dict[int, int]:
        return {1: 6, 2: 7}

    @staticmethod
    def test_fn(*args):
        q = qkv_rope_append70b_op(*args)
        return q, args[6], args[7]

    def test_args(self, case: tuple) -> tuple:
        B, max_seq_len, append_pos = case
        pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        num_pages = B * pages_per_seq

        x = torch.randn(B, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda")
        qkv_weights = (
            torch.randn(1, QKV_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda")
            * (HIDDEN_DIM ** -0.5)
        )
        qkv_weights = interleave_qkv_weights(qkv_weights)
        rope_cos = interleave_rope_table(torch.randn(max_seq_len, HEAD_DIM, dtype=torch.float32, device="cuda"))
        rope_sin = interleave_rope_table(torch.randn(max_seq_len, HEAD_DIM, dtype=torch.float32, device="cuda"))
        pos_ids = torch.full((B,), append_pos, dtype=torch.int32, device="cuda")
        seq_ids = torch.arange(B, dtype=torch.int32, device="cuda")
        append_page = append_pos // PAGE_SIZE
        append_offset = append_pos % PAGE_SIZE
        append_ids = (seq_ids * pages_per_seq + append_page) * PAGE_SIZE + append_offset
        k_cache = torch.zeros(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        v_cache = torch.zeros(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        return x, qkv_weights, rope_cos, rope_sin, pos_ids, append_ids, k_cache, v_cache

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> int:
        out_range = dst_ranges[0]
        B = out_range[-2].size
        return 2 * (B // self.M_INST) * (QKV_DIM // self.Nb)

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        q_range = dst_ranges[0]
        w_range = src_ranges[1]
        k_range = src_ranges[6]
        layer_idx = w_range[-3].start
        base_page = k_range[-4].start
        m_start = q_range[-2].start // self.M_INST
        m_stop = q_range[-2].stop // self.M_INST
        indices = []
        for m in range(m_start, m_stop):
            for n in range(QKV_DIM // self.Nb):
                index = (layer_idx, base_page, m, n)
                indices.append(index)
                indices.append(index)
        return indices

    def access_regions(
        self,
        block_index: Tuple[int, ...],
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ):
        layer_idx, base_page, m, n = block_index
        B = dst_metas[0].shape[-2]
        num_layers = src_metas[1].shape[-3]
        num_pages = src_metas[6].shape[-4] // num_layers
        pages_per_seq = num_pages // B
        layer_page_start = base_page + layer_idx * num_pages
        row_start = m * self.M_INST
        row_stop = row_start + self.M_INST
        n_start = n * self.Nb
        n_stop = n_start + self.Nb

        x_region = ((row_start, row_stop), (0, HIDDEN_DIM))
        w_region = ((layer_idx, layer_idx + 1), (n_start, n_stop), (0, HIDDEN_DIM))
        rope_region = ((0, src_metas[2].shape[-2]), (0, HEAD_DIM))
        ids_region = ((row_start, row_stop),)

        empty_q = ((0, 0), (0, 0))
        empty_kv = ((0, 0), (0, 0), (0, 0), (0, 0))

        q_regions = [empty_q]
        k_regions = [empty_kv]
        v_regions = [empty_kv]

        if n_start < Q_DIM:
            q_regions = [((row_start, row_stop), (n_start, min(n_stop, Q_DIM)))]
        elif n_start < Q_DIM + KV_DIM:
            head_start = (n_start - Q_DIM) // HEAD_DIM
            head_stop = (min(n_stop, Q_DIM + KV_DIM) - Q_DIM + HEAD_DIM - 1) // HEAD_DIM
            page_start = layer_page_start + row_start * pages_per_seq
            page_stop = layer_page_start + row_stop * pages_per_seq
            k_regions = [((page_start, page_stop), (0, PAGE_SIZE), (head_start, head_stop), (0, HEAD_DIM))]
        else:
            head_start = (n_start - Q_DIM - KV_DIM) // HEAD_DIM
            head_stop = (min(n_stop, QKV_DIM) - Q_DIM - KV_DIM + HEAD_DIM - 1) // HEAD_DIM
            page_start = layer_page_start + row_start * pages_per_seq
            page_stop = layer_page_start + row_stop * pages_per_seq
            v_regions = [((page_start, page_stop), (0, PAGE_SIZE), (head_start, head_stop), (0, HEAD_DIM))]

        return (
            [[x_region], [w_region], [rope_region], [rope_region], [ids_region], [ids_region],
             [empty_kv], [empty_kv]],
            [q_regions, k_regions, v_regions],
        )

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)

        x_shape = src_ranges[0].effective_shape
        w_shape = src_ranges[1].effective_shape
        rope_cos_shape = src_ranges[2].effective_shape
        rope_sin_shape = src_ranges[3].effective_shape
        pos_shape = src_ranges[4].effective_shape
        append_shape = src_ranges[5].effective_shape
        k_shape = src_ranges[6].effective_shape
        v_shape = src_ranges[7].effective_shape
        q_shape = dst_ranges[0].effective_shape

        B = x_shape[-2]
        num_layers = src_metas[1].shape[-3]
        total_cache_pages = k_shape[-4]
        if total_cache_pages % num_layers != 0:
            raise RuntimeError(
                f"[MegaKittens] QkvRopeAppend70b requires cache pages divisible by num_layers, "
                f"got cache_pages={total_cache_pages}, num_layers={num_layers}"
            )
        num_pages = total_cache_pages // num_layers

        if x_shape[-1] != HIDDEN_DIM:
            raise RuntimeError(f"[MegaKittens] QkvRopeAppend70b expected x dim={HIDDEN_DIM}, got {x_shape[-1]}")
        if w_shape[-2:] != (QKV_DIM, HIDDEN_DIM):
            raise RuntimeError(
                f"[MegaKittens] QkvRopeAppend70b expected qkv_weights shape (*, {QKV_DIM}, {HIDDEN_DIM}), "
                f"got {w_shape}"
            )
        if rope_cos_shape[-1] != HEAD_DIM or rope_sin_shape[-1] != HEAD_DIM:
            raise RuntimeError(
                f"[MegaKittens] QkvRopeAppend70b expected RoPE tables last dim={HEAD_DIM}, "
                f"got cos={rope_cos_shape}, sin={rope_sin_shape}"
            )
        if pos_shape[-1] != B or append_shape[-1] != B:
            raise RuntimeError(
                f"[MegaKittens] QkvRopeAppend70b expected pos_ids/append_ids shape ({B},), "
                f"got pos={pos_shape}, append={append_shape}"
            )
        if k_shape != v_shape:
            raise RuntimeError(f"[MegaKittens] QkvRopeAppend70b K/V cache shape mismatch: k={k_shape}, v={v_shape}")
        if k_shape[-3:] != (PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM):
            raise RuntimeError(
                f"[MegaKittens] QkvRopeAppend70b expected cache trailing shape "
                f"({PAGE_SIZE}, {NUM_KV_HEADS}, {HEAD_DIM}), got {k_shape[-3:]}"
            )
        if q_shape[-2:] != (B, HIDDEN_DIM):
            raise RuntimeError(
                f"[MegaKittens] QkvRopeAppend70b expected q output shape ({B}, {HIDDEN_DIM}), got {q_shape}"
            )
        if num_pages % B != 0:
            raise RuntimeError(
                f"[MegaKittens] QkvRopeAppend70b requires num_pages divisible by B, got num_pages={num_pages}, B={B}"
            )
        if B % self.M_INST != 0:
            raise RuntimeError(f"[MegaKittens] QkvRopeAppend70b requires B divisible by {self.M_INST}, got B={B}")

        if self.batch_size and B != self.batch_size:
            raise RuntimeError(f"[MegaKittens] QkvRopeAppend70b expected B={self.batch_size}, got {B}")
        if self.num_pages and num_pages != self.num_pages:
            raise RuntimeError(
                f"[MegaKittens] QkvRopeAppend70b expected num_pages={self.num_pages}, got {num_pages}"
            )
