"""Smoke + microbench for the qkv_rope_append_forward binding.

Reference path mirrors megakittens/itypes/llama70b/qkv_rope_append.py:
weights and RoPE tables arrive Q/K-row interleaved (so the fused RoPE in
registers can rotate complex pairs without a shuffle), and append_ids
encode page*PAGE_SIZE + offset within the per-layer paged KV cache.
"""

from __future__ import annotations

import torch

from . import _C


HIDDEN_DIM = 8192
HEAD_DIM = 128
NUM_Q_HEADS = 64
NUM_KV_HEADS = 8
Q_DIM = NUM_Q_HEADS * HEAD_DIM
KV_DIM = NUM_KV_HEADS * HEAD_DIM
QKV_DIM = Q_DIM + 2 * KV_DIM
PAGE_SIZE = 128


def _interleave_indices(num_heads: int, head_dim: int, device=None) -> torch.Tensor:
    half = head_dim // 2
    idx = []
    for h in range(num_heads):
        off = h * head_dim
        for i in range(half):
            idx.append(off + i)
            idx.append(off + half + i)
    return torch.tensor(idx, dtype=torch.long, device=device)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    xe = xf[..., ::2]
    xo = xf[..., 1::2]
    rotated = torch.stack((-xo, xe), dim=-1).flatten(-2)
    return (xf * cos + rotated * sin).to(x.dtype)


def _ref(x, qkv_w, rope_cos, rope_sin, pos_id, append_ids, k_cache, v_cache):
    B = x.shape[0]
    qkv = (x.float() @ qkv_w[0].float().transpose(-1, -2))
    q = qkv[:, :Q_DIM].view(B, NUM_Q_HEADS, HEAD_DIM).to(x.dtype)
    k = qkv[:, Q_DIM:Q_DIM + KV_DIM].view(B, NUM_KV_HEADS, HEAD_DIM).to(x.dtype)
    v = qkv[:, Q_DIM + KV_DIM:].view(B, NUM_KV_HEADS, HEAD_DIM).to(x.dtype)

    pos = pos_id.long().reshape(-1)
    cos = rope_cos[pos].view(-1, 1, HEAD_DIM).expand(B, 1, HEAD_DIM)
    sin = rope_sin[pos].view(-1, 1, HEAD_DIM).expand(B, 1, HEAD_DIM)
    q = _apply_rope(q, cos, sin)
    k = _apply_rope(k, cos, sin)

    k_flat = k_cache.view(-1, NUM_KV_HEADS, HEAD_DIM)
    v_flat = v_cache.view(-1, NUM_KV_HEADS, HEAD_DIM)
    idx = append_ids.long()
    k_flat[idx] = k
    v_flat[idx] = v

    return q.reshape(B, Q_DIM)


def _make_case(B: int, max_seq_len: int, append_pos: int, device="cuda"):
    pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = B * pages_per_seq
    x = torch.randn(B, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    qkv_raw = torch.randn(1, QKV_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device) * (HIDDEN_DIM ** -0.5)
    q_idx = _interleave_indices(NUM_Q_HEADS, HEAD_DIM, device=device)
    k_idx = _interleave_indices(NUM_KV_HEADS, HEAD_DIM, device=device)
    qkv_w = torch.cat((
        qkv_raw[..., :Q_DIM, :][..., q_idx, :],
        qkv_raw[..., Q_DIM:Q_DIM + KV_DIM, :][..., k_idx, :],
        qkv_raw[..., Q_DIM + KV_DIM:, :],
    ), dim=-2).contiguous()
    one_head_idx = _interleave_indices(1, HEAD_DIM, device=device)
    rope_cos = torch.randn(max_seq_len, HEAD_DIM, dtype=torch.float32, device=device)[..., one_head_idx].contiguous()
    rope_sin = torch.randn(max_seq_len, HEAD_DIM, dtype=torch.float32, device=device)[..., one_head_idx].contiguous()
    pos_id = torch.tensor([append_pos], dtype=torch.int32, device=device)
    seq_ids = torch.arange(B, dtype=torch.int32, device=device)
    page = append_pos // PAGE_SIZE
    offset = append_pos % PAGE_SIZE
    append_ids = ((seq_ids * pages_per_seq + page) * PAGE_SIZE + offset).to(torch.int32)
    k_cache = torch.zeros(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    return x, qkv_w, rope_cos, rope_sin, pos_id, append_ids, k_cache, v_cache


def main():
    torch.manual_seed(0)
    cases = [
        (512,  128, 62),
        (1024, 128, 62),
        (1024, 256, 127),
        (2048, 128, 62),
    ]
    for B, max_seq_len, append_pos in cases:
        x, qkv_w, rope_cos, rope_sin, pos_id, append_ids, k_cache, v_cache = _make_case(B, max_seq_len, append_pos)
        k_ref = k_cache.clone()
        v_ref = v_cache.clone()
        q_ref = _ref(x, qkv_w, rope_cos, rope_sin, pos_id, append_ids, k_ref, v_ref)

        q_out = torch.empty(B, Q_DIM, dtype=torch.bfloat16, device="cuda")
        _C.qkv_rope_append_forward(
            x, qkv_w, rope_cos, rope_sin, pos_id, append_ids, k_cache, v_cache, q_out,
        )
        torch.cuda.synchronize()

        q_err = (q_out.float() - q_ref.float()).abs()
        k_err = (k_cache.float() - k_ref.float()).abs()
        v_err = (v_cache.float() - v_ref.float()).abs()

        warmup, iters = 20, 200
        for _ in range(warmup):
            _C.qkv_rope_append_forward(
                x, qkv_w, rope_cos, rope_sin, pos_id, append_ids, k_cache, v_cache, q_out,
            )
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _C.qkv_rope_append_forward(
                x, qkv_w, rope_cos, rope_sin, pos_id, append_ids, k_cache, v_cache, q_out,
            )
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000.0 / iters
        print(f"B={B:5d}  q_err={q_err.max().item():.4g}  k_err={k_err.max().item():.4g}  "
              f"v_err={v_err.max().item():.4g}  {us:7.2f} us")


if __name__ == "__main__":
    main()
