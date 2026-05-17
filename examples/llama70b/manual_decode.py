"""Pure-pytorch Llama-3.3-70B decode using the same paged KV layout and
interleaved RoPE as compiled_decode.py. Acts as the numerics oracle and the
"naive eager pytorch" rung of the megakernel ablation ladder.

Each op is its own small helper so it can be replaced one-at-a-time with a
custom CUDA kernel binding in a follow-up file.
"""

from __future__ import annotations

import argparse

import torch

from .compiled_decode import (
    HIDDEN_DIM,
    HEAD_DIM,
    NUM_Q_HEADS,
    NUM_KV_HEADS,
    GQA_RATIO,
    Q_DIM,
    KV_DIM,
    PAGE_SIZE,
    benchmark_tok_per_sec,
)
from .manual_kernels import _C


NUM_LAYERS = 80


def _rms_kernel(x: torch.Tensor, weight: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    _C.rms_forward(x, weight, eps, out)
    return out


def _qkv_rope_append_kernel(
    hidden_norm: torch.Tensor,
    qkv_w: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    pos_id: torch.Tensor,
    kv_append_indices: torch.Tensor,
    layer_k: torch.Tensor,
    layer_v: torch.Tensor,
) -> torch.Tensor:
    B = hidden_norm.shape[-2]
    q = torch.empty(B, Q_DIM, dtype=hidden_norm.dtype, device=hidden_norm.device)
    _C.qkv_rope_append_forward(
        hidden_norm, qkv_w, rope_cos, rope_sin, pos_id, kv_append_indices, layer_k, layer_v, q,
    )
    return q


def _apply_rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x, cos, and sin are pre-interleaved as [first_half_0, second_half_0, ...].
    x_float = x.float()
    x_even = x_float[..., ::2]
    x_odd = x_float[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    return (x_float * cos + rotated * sin).to(x.dtype)


def _qkv_rope_append_torch(
    hidden_norm: torch.Tensor,
    qkv_w: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    pos_id: torch.Tensor,
    kv_append_indices: torch.Tensor,
    k_cache_layer: torch.Tensor,
    v_cache_layer: torch.Tensor,
) -> torch.Tensor:
    B = hidden_norm.shape[-2]
    qkv = hidden_norm @ qkv_w[0].transpose(-1, -2)
    q = qkv[..., :Q_DIM].view(B, NUM_Q_HEADS, HEAD_DIM)
    k = qkv[..., Q_DIM:Q_DIM + KV_DIM].view(B, NUM_KV_HEADS, HEAD_DIM)
    v = qkv[..., Q_DIM + KV_DIM:].view(B, NUM_KV_HEADS, HEAD_DIM)

    cos = rope_cos[pos_id.long()]
    sin = rope_sin[pos_id.long()]
    q = _apply_rope_torch(q, cos, sin)
    k = _apply_rope_torch(k, cos, sin)

    k_flat = k_cache_layer.view(-1, NUM_KV_HEADS, HEAD_DIM)
    v_flat = v_cache_layer.view(-1, NUM_KV_HEADS, HEAD_DIM)
    idx = kv_append_indices.long()
    k_flat[idx] = k
    v_flat[idx] = v

    return q.reshape(B, HIDDEN_DIM)


def _attention_torch(
    q_flat: torch.Tensor,
    k_cache_layer: torch.Tensor,
    v_cache_layer: torch.Tensor,
    pos_id: torch.Tensor,
    attn_scale: torch.Tensor,
) -> torch.Tensor:
    B = q_flat.shape[-2]
    num_pages = k_cache_layer.shape[0]
    pages_per_seq = num_pages // B
    seq_len = pages_per_seq * PAGE_SIZE

    q = q_flat.view(B, NUM_Q_HEADS, HEAD_DIM).float()
    k_seq = k_cache_layer.view(B, seq_len, NUM_KV_HEADS, HEAD_DIM).float()
    v_seq = v_cache_layer.view(B, seq_len, NUM_KV_HEADS, HEAD_DIM).float()

    q_grouped = q.view(B, NUM_KV_HEADS, GQA_RATIO, HEAD_DIM)
    k_t = k_seq.permute(0, 2, 3, 1)
    scores = torch.matmul(q_grouped, k_t) * attn_scale

    positions = torch.arange(seq_len, device=q_flat.device, dtype=pos_id.dtype)
    mask = (positions <= pos_id).view(1, 1, 1, seq_len)
    scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    v_perm = v_seq.permute(0, 2, 1, 3)
    out = torch.matmul(weights, v_perm)
    return out.reshape(B, NUM_Q_HEADS * HEAD_DIM).to(q_flat.dtype)


def _o_proj_residual_kernel(hidden: torch.Tensor, attn_out: torch.Tensor, o_w: torch.Tensor) -> None:
    _C.o_proj_residual_forward(attn_out, o_w, hidden)


def _gate_silu_kernel(x: torch.Tensor, gate_w: torch.Tensor) -> torch.Tensor:
    M, _ = x.shape
    N = gate_w.shape[-2]
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    _C.gate_silu_forward(x, gate_w, out)
    return out


def _up_matmul_kernel(x: torch.Tensor, up_w: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    M, _ = x.shape
    N = up_w.shape[-2]
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    _C.up_matmul_forward(x, up_w, gate, out)
    return out


def _attention_decode_kernel(
    q_flat: torch.Tensor,
    k_cache_layer: torch.Tensor,
    v_cache_layer: torch.Tensor,
    pos_id: torch.Tensor,
    attn_scale: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(q_flat)
    _C.attention_decode_forward(q_flat, k_cache_layer, v_cache_layer, pos_id, attn_scale, out)
    return out


def _lm_head_kernel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    M, _ = x.shape
    N = w.shape[-2]
    logits = torch.empty(M, N, dtype=x.dtype, device=x.device)
    _C.lm_head_forward(x, w, logits)
    return logits


def decode(
    hidden,                # [B, HIDDEN_DIM] bf16
    qkv_weights,           # [L, QKV_DIM, HIDDEN_DIM] bf16, Q/K interleaved
    o_weights,             # [L, HIDDEN_DIM, HIDDEN_DIM] bf16
    attn_norm_weights,     # [L, HIDDEN_DIM] bf16
    mlp_norm_weights,      # [L, HIDDEN_DIM] bf16
    gate_weights,          # [L, INTERMEDIATE_DIM, HIDDEN_DIM] bf16
    up_weights,            # [L, INTERMEDIATE_DIM, HIDDEN_DIM] bf16
    down_weights,          # [L, HIDDEN_DIM, INTERMEDIATE_DIM] bf16
    lm_head_norm_weight,   # [1, HIDDEN_DIM] bf16
    lm_head_weight,        # [1, VOCAB_SIZE, HIDDEN_DIM] bf16
    k_cache,               # [L * num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM] bf16
    v_cache,               # [L * num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM] bf16
    rope_cos,              # [max_seq_len, HEAD_DIM] fp32, interleaved
    rope_sin,              # [max_seq_len, HEAD_DIM] fp32, interleaved
    kv_append_indices,     # [B] int32, page * PAGE_SIZE + offset in layer-local page space
    pos_id,                # [1] int32
    attn_scale,            # [1] fp32
    rms_norm_eps,          # [1] fp32
):
    num_pages = k_cache.shape[-4] // NUM_LAYERS

    for layer_idx in range(NUM_LAYERS):
        hidden_norm = _rms_kernel(hidden, attn_norm_weights[layer_idx], rms_norm_eps)
        layer_page_start = layer_idx * num_pages
        layer_page_stop = layer_page_start + num_pages
        layer_k = k_cache[layer_page_start:layer_page_stop]
        layer_v = v_cache[layer_page_start:layer_page_stop]
        q = _qkv_rope_append_kernel(
            hidden_norm,
            qkv_weights[layer_idx:layer_idx + 1],
            rope_cos,
            rope_sin,
            pos_id,
            kv_append_indices,
            layer_k,
            layer_v,
        )

        attn_out = _attention_decode_kernel(q, layer_k, layer_v, pos_id, attn_scale)

        _o_proj_residual_kernel(hidden, attn_out, o_weights[layer_idx:layer_idx + 1])

        mlp_norm = _rms_kernel(hidden, mlp_norm_weights[layer_idx], rms_norm_eps)
        gate = _gate_silu_kernel(mlp_norm, gate_weights[layer_idx:layer_idx + 1])
        up = _up_matmul_kernel(mlp_norm, up_weights[layer_idx:layer_idx + 1], gate)
        _o_proj_residual_kernel(hidden, up, down_weights[layer_idx:layer_idx + 1])

    logits_hidden = _rms_kernel(hidden, lm_head_norm_weight[0], rms_norm_eps)
    logits = _lm_head_kernel(logits_hidden, lm_head_weight)
    pos_id.add_(1)
    return logits


class GraphedDecode:
    # Positional indices into the decode() arg tuple.
    K_CACHE_IDX = 10
    V_CACHE_IDX = 11
    POS_ID_IDX = 15

    def __init__(self, decode_fn, warmup_iters: int = 3):
        self.decode_fn = decode_fn
        self.warmup_iters = warmup_iters
        self.graph = None
        self.logits = None

    def __call__(self, *args):
        if self.graph is None:
            k_cache = args[self.K_CACHE_IDX]
            v_cache = args[self.V_CACHE_IDX]
            pos_id = args[self.POS_ID_IDX]

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(self.warmup_iters):
                    self.decode_fn(*args)
            torch.cuda.current_stream().wait_stream(s)

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.logits = self.decode_fn(*args)

            # Warmup + capture advanced pos_id and wrote garbage into kv;
            # reset so the triggering call returns from a clean state.
            pos_id.zero_()
            k_cache.zero_()
            v_cache.zero_()
            return self.logits

        self.graph.replay()
        return self.logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Hello, my name is")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--cuda-graph", action="store_true",
                        help="Wrap decode in a CUDA graph (capture once, replay per token).")
    args = parser.parse_args()

    NUM_LAYERS = args.num_layers

    decode_fn = GraphedDecode(decode) if args.cuda_graph else decode

    benchmark_tok_per_sec(
        decode_fn,
        num_layers=NUM_LAYERS,
        prompt=args.prompt,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        warmup=args.warmup,
    )
