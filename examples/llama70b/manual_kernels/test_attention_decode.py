"""Smoke + microbench for the attention_decode_forward binding.

Paged GQA flash-attention decode. Reference is the same loop as
megakittens/itypes/llama70b/attention_decode.py.
"""

from __future__ import annotations

try:
    import pytest
except ImportError:
    pytest = None

import torch

from . import _C


HIDDEN_DIM    = 8192
HEAD_DIM      = 128
NUM_Q_HEADS   = 64
NUM_KV_HEADS  = 8
GQA_RATIO     = NUM_Q_HEADS // NUM_KV_HEADS
PAGE_SIZE     = 128


def _ref(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
         pos_id: torch.Tensor, attn_scale: torch.Tensor) -> torch.Tensor:
    B = q.shape[0]
    num_pages = k_cache.shape[0]
    pages_per_seq = num_pages // B
    seq_len = int(pos_id.item()) + 1
    num_pages_to_read = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    scale = float(attn_scale.item())

    q_heads = q.view(B, NUM_Q_HEADS, HEAD_DIM).float()
    out = torch.empty_like(q_heads)
    for seq_idx in range(B):
        page_start = seq_idx * pages_per_seq
        k_hist = k_cache[page_start:page_start + num_pages_to_read].reshape(
            -1, NUM_KV_HEADS, HEAD_DIM)[:seq_len].float()
        v_hist = v_cache[page_start:page_start + num_pages_to_read].reshape(
            -1, NUM_KV_HEADS, HEAD_DIM)[:seq_len].float()
        q_grouped = q_heads[seq_idx].view(NUM_KV_HEADS, GQA_RATIO, HEAD_DIM)
        scores  = torch.einsum("kgd,tkd->kgt", q_grouped, k_hist) * scale
        weights = torch.softmax(scores, dim=-1)
        out[seq_idx] = torch.einsum("kgt,tkd->kgd", weights, v_hist).reshape(
            NUM_Q_HEADS, HEAD_DIM)
    return out.reshape(B, HIDDEN_DIM).to(q.dtype)


def _make_case(B: int, max_seq_len: int, pos: int, device: str = "cuda"):
    pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = B * pages_per_seq
    q = torch.randn(B, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                          dtype=torch.bfloat16, device=device)
    v_cache = torch.randn(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                          dtype=torch.bfloat16, device=device)
    pos_id = torch.tensor([pos], dtype=torch.int32, device=device)
    attn_scale = torch.tensor([1.0 / (HEAD_DIM ** 0.5)],
                              dtype=torch.float32, device=device)
    return q, k_cache, v_cache, pos_id, attn_scale


_CASES = [
    (128,  128, 15),
    (128,  128, 62),
    (128,  128, 63),
    (256,  256, 127),
    (512,  512, 255),
    (1024, 128, 127),
    (1024, 256, 255),
]


if pytest is not None:
    @pytest.mark.parametrize("B,max_seq_len,pos", _CASES)
    def test_attention_decode(B, max_seq_len, pos):
        torch.manual_seed(0)
        q, kc, vc, pid, scale = _make_case(B, max_seq_len, pos)
        ref = _ref(q, kc, vc, pid, scale)
        out = torch.empty_like(q)
        _C.attention_decode_forward(q, kc, vc, pid, scale, out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def main():
    torch.manual_seed(0)
    for B, max_seq_len, pos in _CASES:
        q, kc, vc, pid, scale = _make_case(B, max_seq_len, pos)
        ref = _ref(q, kc, vc, pid, scale)
        out = torch.empty_like(q)
        _C.attention_decode_forward(q, kc, vc, pid, scale, out)
        torch.cuda.synchronize()
        err = (out.float() - ref.float()).abs()

        warmup, iters = 20, 200
        for _ in range(warmup):
            _C.attention_decode_forward(q, kc, vc, pid, scale, out)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _C.attention_decode_forward(q, kc, vc, pid, scale, out)
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000.0 / iters
        seq_len = pos + 1
        # flop count: 2 * B * NUM_Q_HEADS * HEAD_DIM * seq_len * 2 (QK and PV)
        tflops = 4.0 * B * NUM_Q_HEADS * HEAD_DIM * seq_len / (us * 1e-6) / 1e12
        print(f"B={B:5d} max_seq={max_seq_len:4d} pos={pos:4d}  "
              f"err={err.max().item():.4g}  {us:7.2f} us  {tflops:6.2f} TFLOPS")


if __name__ == "__main__":
    main()
