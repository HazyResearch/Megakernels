"""1-layer decode using megakittens.compile on explicit fused llama1b ops.

Step 1 of the explicit-ops-with-auto-schedule experiment: write the decode
forward in plain PyTorch using the five fused custom ops, slicing per-layer
weights via `[i:i+1]` so the tracer narrows each op's TensorRange. Then
`megakittens.compile` runs the tracer + scheduler + dispatcher end-to-end
instead of the hand-written schedule in `scheduler.py`.

At 1 layer the hardcoded `layer_idx=0` in each itype's `block_indices` happens
to be correct, so this is a safe first rung. Scaling to multi-layer will
require:
  1. block_indices to read layer from src_ranges (not hardcode 0)
  2. rms_qkv_rope_append to declare mutates_args for k_cache/v_cache
"""

from __future__ import annotations

import math

import torch

import megakittens
from megakittens.jit.cuda_utils import initialize_cuda_context
from .scheduler import (
    HEAD_DIM,
    HIDDEN_DIM,
    INTERMEDIATE_DIM,
    MAX_SEQ_LEN,
    NUM_KV_HEADS,
    QKV_DIM,
    RMS_NORM_EPS,
    VOCAB_SIZE,
)

ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)


def decode_one_layer(
    hidden_states,         # [HIDDEN_DIM] bf16
    qkv_weights,           # [L, QKV_DIM, HIDDEN_DIM] bf16
    o_weights,             # [L, HIDDEN_DIM, HIDDEN_DIM] bf16
    attn_norm_weights,     # [L, HIDDEN_DIM] bf16
    mlp_norm_weights,      # [L, HIDDEN_DIM] bf16
    up_weights,            # [L, INTERMEDIATE_DIM, HIDDEN_DIM] bf16
    gate_weights,          # [L, INTERMEDIATE_DIM, HIDDEN_DIM] bf16
    down_weights,          # [L, HIDDEN_DIM, INTERMEDIATE_DIM] bf16
    lm_head_norm_weight,   # [HIDDEN_DIM] bf16
    lm_head_weight,        # [VOCAB_SIZE, HIDDEN_DIM] bf16
    k_cache,               # [L, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM] bf16
    v_cache,               # [L, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM] bf16
    rope_cos,              # [MAX_SEQ_LEN, HEAD_DIM] fp32
    rope_sin,              # [MAX_SEQ_LEN, HEAD_DIM] fp32
    pos_id,                # [1] int32
    attn_scale,            # [1] fp32
    rms_norm_eps,          # [1] fp32
):
    i = 0  # single-layer forward; `[i:i+1]` on stacked tensors keeps layer rank

    q = torch.ops.megakittens.rms_qkv_rope_append(
        hidden_states,
        attn_norm_weights[i:i+1],
        qkv_weights[i:i+1],
        rope_cos,
        rope_sin,
        k_cache[i:i+1],
        v_cache[i:i+1],
        pos_id,
        rms_norm_eps,
    )

    attn_out = torch.ops.megakittens.attention_partial(
        q, k_cache[i:i+1], v_cache[i:i+1], pos_id, attn_scale,
    )

    # o_proj + residual: mutates hidden_states in place
    torch.ops.megakittens.mat_vec_adds(hidden_states, attn_out, o_weights[i:i+1])

    silu_out = torch.ops.megakittens.rms_upgate_silu(
        hidden_states,
        mlp_norm_weights[i:i+1],
        up_weights[i:i+1],
        gate_weights[i:i+1],
        rms_norm_eps,
    )

    # down_proj + residual: mutates hidden_states in place
    torch.ops.megakittens.mat_vec_adds(hidden_states, silu_out, down_weights[i:i+1])

    logits = torch.ops.megakittens.rms_lm_head(
        hidden_states, lm_head_norm_weight, lm_head_weight, rms_norm_eps,
    )
    return logits


def main():
    initialize_cuda_context()
    torch._dynamo.reset()

    D = "cuda"
    L = 1  # number of layers for this smoke test

    hidden = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    qkv_w = torch.randn(L, QKV_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    o_w = torch.randn(L, HIDDEN_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    attn_norm = torch.randn(L, HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    mlp_norm = torch.randn(L, HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    up_w = torch.randn(L, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    gate_w = torch.randn(L, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    down_w = torch.randn(L, HIDDEN_DIM, INTERMEDIATE_DIM, dtype=torch.bfloat16, device=D)
    lm_head_norm = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    lm_head = torch.randn(VOCAB_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    k_cache = torch.zeros(L, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    v_cache = torch.zeros(L, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    rope_cos = torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=D)
    rope_sin = torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=D)
    pos_id = torch.tensor([0], dtype=torch.int32, device=D)
    attn_scale = torch.tensor([ATTN_SCALE], dtype=torch.float32, device=D)
    rms_eps = torch.tensor([RMS_NORM_EPS], dtype=torch.float32, device=D)

    args = (
        hidden, qkv_w, o_w, attn_norm, mlp_norm, up_w, gate_w, down_w,
        lm_head_norm, lm_head, k_cache, v_cache, rope_cos, rope_sin,
        pos_id, attn_scale, rms_eps,
    )

    # Eager reference: run the uncompiled forward first to capture expected logits.
    eager_hidden = hidden.clone()
    eager_k = k_cache.clone()
    eager_v = v_cache.clone()
    eager_args = (
        eager_hidden, qkv_w, o_w, attn_norm, mlp_norm, up_w, gate_w, down_w,
        lm_head_norm, lm_head, eager_k, eager_v, rope_cos, rope_sin,
        pos_id, attn_scale, rms_eps,
    )
    eager_logits = decode_one_layer(*eager_args)
    torch.cuda.synchronize()

    compiled = megakittens.compile(
        decode_one_layer,
        use_jit_cache=False,
        save_dag=True,
        save_schedule=True,
        verbose=True,
    )
    mk_logits = compiled(*args)
    torch.cuda.synchronize()

    diff = (eager_logits.float() - mk_logits.float()).abs()
    print(f"logits shape: {mk_logits.shape} dtype: {mk_logits.dtype}")
    print(f"max_diff={diff.max().item():.4f}  mean_diff={diff.mean().item():.6f}  "
          f"argmax_match={eager_logits.argmax().item() == mk_logits.argmax().item()}")


if __name__ == "__main__":
    main()
