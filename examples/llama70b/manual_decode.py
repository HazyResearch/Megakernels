"""Llama-3.3-70B decode using regular TK kernels, as an ablation baseline against the megakernel."""

from __future__ import annotations

import argparse
import os

import torch

from .compiled_decode import Q_DIM, benchmark_tok_per_sec
from .manual_kernels import _C


NUM_LAYERS = 80

# Pre-allocated events so timing works inside CUDA graph capture. TIME_KERNELS=1 to enable.
TIME_KERNELS = os.environ.get("TIME_KERNELS") == "1"
_timing_events: dict | None = None


def _init_timing(num_layers: int) -> None:
    global _timing_events
    if not TIME_KERNELS:
        return
    _timing_events = {
        "o_proj_s": [torch.cuda.Event(enable_timing=True) for _ in range(num_layers)],
        "o_proj_e": [torch.cuda.Event(enable_timing=True) for _ in range(num_layers)],
        "down_s":   [torch.cuda.Event(enable_timing=True) for _ in range(num_layers)],
        "down_e":   [torch.cuda.Event(enable_timing=True) for _ in range(num_layers)],
    }


def _report_timing() -> None:
    if _timing_events is None:
        return
    torch.cuda.synchronize()
    n = len(_timing_events["o_proj_s"])
    o = [_timing_events["o_proj_s"][i].elapsed_time(_timing_events["o_proj_e"][i]) for i in range(n)]
    d = [_timing_events["down_s"][i].elapsed_time(_timing_events["down_e"][i]) for i in range(n)]
    print()
    print(f"Per-layer in-decode kernel timings (mean over {n} layers, final replay):")
    print(f"  o_proj : {sum(o)/n*1000:7.2f} us   (min {min(o)*1000:.2f}, max {max(o)*1000:.2f})")
    print(f"  down   : {sum(d)/n*1000:7.2f} us   (min {min(d)*1000:.2f}, max {max(d)*1000:.2f})")


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
    kv_append_indices,     # [B] int32, layer-local kv slot index
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

        if TIME_KERNELS: _timing_events["o_proj_s"][layer_idx].record()
        _o_proj_residual_kernel(hidden, attn_out, o_weights[layer_idx:layer_idx + 1])
        if TIME_KERNELS: _timing_events["o_proj_e"][layer_idx].record()

        mlp_norm = _rms_kernel(hidden, mlp_norm_weights[layer_idx], rms_norm_eps)
        gate = _gate_silu_kernel(mlp_norm, gate_weights[layer_idx:layer_idx + 1])
        up = _up_matmul_kernel(mlp_norm, up_weights[layer_idx:layer_idx + 1], gate)
        if TIME_KERNELS: _timing_events["down_s"][layer_idx].record()
        _o_proj_residual_kernel(hidden, up, down_weights[layer_idx:layer_idx + 1])
        if TIME_KERNELS: _timing_events["down_e"][layer_idx].record()

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

            # Undo state mutated during warmup + capture.
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

    _init_timing(NUM_LAYERS)
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
    _report_timing()
