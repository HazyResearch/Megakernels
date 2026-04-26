"""Step-by-step bisection of compiled Llama-70B decode against eager torch.

Builds up the first decoder layer one instruction at a time and compares the
megakernel-compiled output against a pure-eager reference at each step. The
first step where outputs diverge identifies which IType is incorrect when fed
real upstream activations (real HF weights + real prompt embeddings).

Usage:
    python -m examples.llama70b.bisect_decode --step 1
"""

from __future__ import annotations

import argparse
import math

import torch

from examples.llama70b.compiled_decode import (
    HIDDEN_DIM, RMS_NORM_EPS, MODEL_ID, PAGE_SIZE, NUM_LAYERS,
    NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, Q_DIM, KV_DIM,
    load_hf_weights, _kv_append_indices,
)
from megakittens.itypes.llama70b.qkv_rope_append import _apply_rope
from megakittens.itypes.llama70b.qkv_rope_append import qkv_rope_append70b_op as qkv_rope_append70b_op_eager_call
from megakittens.jit.cuda_utils import initialize_cuda_context
from transformers import AutoTokenizer
import megakittens


def step_1_compiled(hidden, attn_norm_weight, eps):
    """Step 1: only the first rms (attn norm, layer 0) under megakittens.compile."""
    return torch.ops.megakittens.rms70b(hidden, attn_norm_weight, eps)


def step_1_eager(hidden, attn_norm_weight, eps):
    """Step 1: pure-eager reference (no custom op)."""
    return torch.rms_norm(hidden, [hidden.shape[-1]], attn_norm_weight, eps.item())


def step_2_compiled(hidden, attn_norm_weight, eps,
                    qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
                    k_cache, v_cache):
    """Step 2: rms + qkv_rope_append (layer 0)."""
    h_norm = torch.ops.megakittens.rms70b(hidden, attn_norm_weight, eps)
    q = torch.ops.megakittens.qkv_rope_append70b(
        h_norm, qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices, k_cache, v_cache,
    )
    return q


def step_2_eager(hidden, attn_norm_weight, eps,
                 qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
                 k_cache, v_cache):
    """Step 2: pure-eager reference."""
    B = hidden.shape[-2]
    h_norm = torch.rms_norm(hidden, [hidden.shape[-1]], attn_norm_weight, eps.item())
    qkv = h_norm.float() @ qkv_weights[0].float().transpose(-1, -2)
    q = qkv[..., :Q_DIM].view(B, NUM_Q_HEADS, HEAD_DIM).to(hidden.dtype)
    k = qkv[..., Q_DIM:Q_DIM + KV_DIM].view(B, NUM_KV_HEADS, HEAD_DIM).to(hidden.dtype)
    v = qkv[..., Q_DIM + KV_DIM:].view(B, NUM_KV_HEADS, HEAD_DIM).to(hidden.dtype)
    cos = rope_cos[pos_ids.long()].view(B, 1, HEAD_DIM)
    sin = rope_sin[pos_ids.long()].view(B, 1, HEAD_DIM)
    q = _apply_rope(q, cos, sin)
    k = _apply_rope(k, cos, sin)
    for seq_idx in range(B):
        append_idx = int(kv_indices[seq_idx].item())
        page_idx = append_idx // PAGE_SIZE
        page_offset = append_idx % PAGE_SIZE
        k_cache[page_idx, page_offset] = k[seq_idx]
        v_cache[page_idx, page_offset] = v[seq_idx]
    return q.reshape(B, HIDDEN_DIM).to(hidden.dtype)


def step_3_compiled(hidden, attn_norm_weight, eps,
                    qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
                    k_cache, v_cache, pos_id, attn_scale):
    """Step 3: rms + qkv_rope_append + attention_decode (layer 0)."""
    h_norm = torch.ops.megakittens.rms70b(hidden, attn_norm_weight, eps)
    q = torch.ops.megakittens.qkv_rope_append70b(
        h_norm, qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices, k_cache, v_cache,
    )
    attn_out = torch.ops.megakittens.attention_decode70b(q, k_cache, v_cache, pos_id, attn_scale)
    return attn_out


def step_3_eager(hidden, attn_norm_weight, eps,
                 qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
                 k_cache, v_cache, pos_id, attn_scale):
    """Step 3: pure-eager reference (rms + qkv+rope+append + attention)."""
    q = step_2_eager(
        hidden, attn_norm_weight, eps,
        qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices, k_cache, v_cache,
    )
    B = hidden.shape[-2]
    num_pages = k_cache.shape[-4]
    pages_per_seq = num_pages // B
    seq_len = pos_id.item() + 1
    num_pages_to_read = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    GQA_RATIO = NUM_Q_HEADS // NUM_KV_HEADS

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
    return out.reshape(B, HIDDEN_DIM).to(hidden.dtype)


def step_4_compiled(hidden, attn_norm_weight, eps,
                    qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
                    k_cache, v_cache, pos_id, attn_scale, o_weights):
    """Step 4: + oproj_residual (attention) — mutates hidden in place."""
    h_norm = torch.ops.megakittens.rms70b(hidden, attn_norm_weight, eps)
    q = torch.ops.megakittens.qkv_rope_append70b(
        h_norm, qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices, k_cache, v_cache,
    )
    attn_out = torch.ops.megakittens.attention_decode70b(q, k_cache, v_cache, pos_id, attn_scale)
    torch.ops.megakittens.oproj_residual70b(hidden, attn_out, o_weights)
    return hidden


def step_4_eager(hidden, attn_norm_weight, eps,
                 qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
                 k_cache, v_cache, pos_id, attn_scale, o_weights):
    """Step 4: pure-eager reference."""
    attn_out = step_3_eager(
        hidden, attn_norm_weight, eps,
        qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
        k_cache, v_cache, pos_id, attn_scale,
    )
    hidden.add_(attn_out @ o_weights[0].transpose(-1, -2))
    return hidden


def step_5_compiled(hidden, attn_norm_weight, eps,
                    qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
                    k_cache, v_cache, pos_id, attn_scale, o_weights, mlp_norm_weight):
    """Step 5: + rms (mlp norm)."""
    h_norm = torch.ops.megakittens.rms70b(hidden, attn_norm_weight, eps)
    q = torch.ops.megakittens.qkv_rope_append70b(
        h_norm, qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices, k_cache, v_cache,
    )
    attn_out = torch.ops.megakittens.attention_decode70b(q, k_cache, v_cache, pos_id, attn_scale)
    torch.ops.megakittens.oproj_residual70b(hidden, attn_out, o_weights)
    mlp_norm = torch.ops.megakittens.rms70b(hidden, mlp_norm_weight, eps)
    return mlp_norm


def step_5_eager(hidden, attn_norm_weight, eps,
                 qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
                 k_cache, v_cache, pos_id, attn_scale, o_weights, mlp_norm_weight):
    hidden = step_4_eager(
        hidden, attn_norm_weight, eps,
        qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
        k_cache, v_cache, pos_id, attn_scale, o_weights,
    )
    return torch.rms_norm(hidden, [hidden.shape[-1]], mlp_norm_weight, eps.item())


def step_6_compiled(hidden, attn_norm_weight, eps,
                    qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
                    k_cache, v_cache, pos_id, attn_scale, o_weights,
                    mlp_norm_weight, gate_weights):
    """Step 6: + gate_silu."""
    h_norm = torch.ops.megakittens.rms70b(hidden, attn_norm_weight, eps)
    q = torch.ops.megakittens.qkv_rope_append70b(
        h_norm, qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices, k_cache, v_cache,
    )
    attn_out = torch.ops.megakittens.attention_decode70b(q, k_cache, v_cache, pos_id, attn_scale)
    torch.ops.megakittens.oproj_residual70b(hidden, attn_out, o_weights)
    mlp_norm = torch.ops.megakittens.rms70b(hidden, mlp_norm_weight, eps)
    gate = torch.ops.megakittens.gate_silu70b(mlp_norm, gate_weights)
    return gate


def step_6_eager(hidden, attn_norm_weight, eps,
                 qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
                 k_cache, v_cache, pos_id, attn_scale, o_weights,
                 mlp_norm_weight, gate_weights):
    mlp_norm = step_5_eager(
        hidden, attn_norm_weight, eps,
        qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
        k_cache, v_cache, pos_id, attn_scale, o_weights, mlp_norm_weight,
    )
    return torch.nn.functional.silu(mlp_norm @ gate_weights[0].transpose(-1, -2))


def step_7_compiled(hidden, attn_norm_weight, eps,
                    qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
                    k_cache, v_cache, pos_id, attn_scale, o_weights,
                    mlp_norm_weight, gate_weights, up_weights):
    """Step 7: + up_matmul."""
    h_norm = torch.ops.megakittens.rms70b(hidden, attn_norm_weight, eps)
    q = torch.ops.megakittens.qkv_rope_append70b(
        h_norm, qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices, k_cache, v_cache,
    )
    attn_out = torch.ops.megakittens.attention_decode70b(q, k_cache, v_cache, pos_id, attn_scale)
    torch.ops.megakittens.oproj_residual70b(hidden, attn_out, o_weights)
    mlp_norm = torch.ops.megakittens.rms70b(hidden, mlp_norm_weight, eps)
    gate = torch.ops.megakittens.gate_silu70b(mlp_norm, gate_weights)
    up = torch.ops.megakittens.up_matmul70b(mlp_norm, up_weights, gate)
    return up


def step_7_eager(hidden, attn_norm_weight, eps,
                 qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
                 k_cache, v_cache, pos_id, attn_scale, o_weights,
                 mlp_norm_weight, gate_weights, up_weights):
    mlp_norm = step_5_eager(
        hidden, attn_norm_weight, eps,
        qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
        k_cache, v_cache, pos_id, attn_scale, o_weights, mlp_norm_weight,
    )
    gate = torch.nn.functional.silu(mlp_norm @ gate_weights[0].transpose(-1, -2))
    return (mlp_norm @ up_weights[0].transpose(-1, -2)) * gate


def step_8_compiled(hidden, attn_norm_weight, eps,
                    qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
                    k_cache, v_cache, pos_id, attn_scale, o_weights,
                    mlp_norm_weight, gate_weights, up_weights, down_weights):
    """Step 8: full layer 0 (+ oproj_residual for down)."""
    h_norm = torch.ops.megakittens.rms70b(hidden, attn_norm_weight, eps)
    q = torch.ops.megakittens.qkv_rope_append70b(
        h_norm, qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices, k_cache, v_cache,
    )
    attn_out = torch.ops.megakittens.attention_decode70b(q, k_cache, v_cache, pos_id, attn_scale)
    torch.ops.megakittens.oproj_residual70b(hidden, attn_out, o_weights)
    mlp_norm = torch.ops.megakittens.rms70b(hidden, mlp_norm_weight, eps)
    gate = torch.ops.megakittens.gate_silu70b(mlp_norm, gate_weights)
    up = torch.ops.megakittens.up_matmul70b(mlp_norm, up_weights, gate)
    torch.ops.megakittens.oproj_residual70b(hidden, up, down_weights)
    return hidden


def step_8_eager(hidden, attn_norm_weight, eps,
                 qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
                 k_cache, v_cache, pos_id, attn_scale, o_weights,
                 mlp_norm_weight, gate_weights, up_weights, down_weights):
    up = step_7_eager(
        hidden, attn_norm_weight, eps,
        qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
        k_cache, v_cache, pos_id, attn_scale, o_weights,
        mlp_norm_weight, gate_weights, up_weights,
    )
    hidden.add_(up @ down_weights[0].transpose(-1, -2))
    return hidden


def step_9_compiled(hidden, attn_norm_weight, eps,
                    qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
                    k_cache, v_cache, pos_id, attn_scale, o_weights,
                    mlp_norm_weight, gate_weights, up_weights, down_weights,
                    lm_head_norm_weight, lm_head_weight):
    """Step 9: full layer 0 + final rms + lm_head (skip layers 1-79)."""
    h_norm = torch.ops.megakittens.rms70b(hidden, attn_norm_weight, eps)
    q = torch.ops.megakittens.qkv_rope_append70b(
        h_norm, qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices, k_cache, v_cache,
    )
    attn_out = torch.ops.megakittens.attention_decode70b(q, k_cache, v_cache, pos_id, attn_scale)
    torch.ops.megakittens.oproj_residual70b(hidden, attn_out, o_weights)
    mlp_norm = torch.ops.megakittens.rms70b(hidden, mlp_norm_weight, eps)
    gate = torch.ops.megakittens.gate_silu70b(mlp_norm, gate_weights)
    up = torch.ops.megakittens.up_matmul70b(mlp_norm, up_weights, gate)
    torch.ops.megakittens.oproj_residual70b(hidden, up, down_weights)
    final_norm = torch.ops.megakittens.rms70b(hidden, lm_head_norm_weight[0], eps)
    return torch.ops.megakittens.lm_head70b(final_norm, lm_head_weight)


def step_9_eager(hidden, attn_norm_weight, eps,
                 qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
                 k_cache, v_cache, pos_id, attn_scale, o_weights,
                 mlp_norm_weight, gate_weights, up_weights, down_weights,
                 lm_head_norm_weight, lm_head_weight):
    hidden = step_8_eager(
        hidden, attn_norm_weight, eps,
        qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
        k_cache, v_cache, pos_id, attn_scale, o_weights,
        mlp_norm_weight, gate_weights, up_weights, down_weights,
    )
    final_norm = torch.rms_norm(hidden, [hidden.shape[-1]], lm_head_norm_weight[0], eps.item())
    return final_norm @ lm_head_weight[0].transpose(-1, -2)


def step_10_compiled(hidden, qkv_weights, o_weights, attn_norm_weights, mlp_norm_weights,
                     gate_weights, up_weights, down_weights,
                     lm_head_norm_weight, lm_head_weight,
                     k_cache, v_cache, rope_cos, rope_sin,
                     pos_ids, kv_indices, pos_id, attn_scale, eps,
                     num_layers):
    """Step 10: chain `num_layers` layers + final rms + lm_head."""
    batch_size = hidden.shape[-2]
    # k_cache is allocated for the full NUM_LAYERS layout, so divide by NUM_LAYERS
    # (not by `num_layers` arg, which is just how many we *run*).
    num_pages = k_cache.shape[-4] // NUM_LAYERS
    for layer_idx in range(num_layers):
        h_norm = torch.ops.megakittens.rms70b(hidden, attn_norm_weights[layer_idx], eps)
        layer_k = k_cache[layer_idx * num_pages:(layer_idx + 1) * num_pages]
        layer_v = v_cache[layer_idx * num_pages:(layer_idx + 1) * num_pages]
        q = torch.ops.megakittens.qkv_rope_append70b(
            h_norm, qkv_weights[layer_idx:layer_idx + 1], rope_cos, rope_sin,
            pos_ids, kv_indices, layer_k, layer_v,
        )
        attn_out = torch.ops.megakittens.attention_decode70b(q, layer_k, layer_v, pos_id, attn_scale)
        torch.ops.megakittens.oproj_residual70b(hidden, attn_out, o_weights[layer_idx:layer_idx + 1])
        mlp_norm = torch.ops.megakittens.rms70b(hidden, mlp_norm_weights[layer_idx], eps)
        gate = torch.ops.megakittens.gate_silu70b(mlp_norm, gate_weights[layer_idx:layer_idx + 1])
        up = torch.ops.megakittens.up_matmul70b(mlp_norm, up_weights[layer_idx:layer_idx + 1], gate)
        torch.ops.megakittens.oproj_residual70b(hidden, up, down_weights[layer_idx:layer_idx + 1])
    final_norm = torch.ops.megakittens.rms70b(hidden, lm_head_norm_weight[0], eps)
    return torch.ops.megakittens.lm_head70b(final_norm, lm_head_weight)


def step_10_eager(hidden, qkv_weights, o_weights, attn_norm_weights, mlp_norm_weights,
                  gate_weights, up_weights, down_weights,
                  lm_head_norm_weight, lm_head_weight,
                  k_cache, v_cache, rope_cos, rope_sin,
                  pos_ids, kv_indices, pos_id, attn_scale, eps,
                  num_layers):
    batch_size = hidden.shape[-2]
    num_pages = k_cache.shape[-4] // NUM_LAYERS
    for layer_idx in range(num_layers):
        layer_k = k_cache[layer_idx * num_pages:(layer_idx + 1) * num_pages]
        layer_v = v_cache[layer_idx * num_pages:(layer_idx + 1) * num_pages]
        hidden = step_8_eager(
            hidden, attn_norm_weights[layer_idx], eps,
            qkv_weights[layer_idx:layer_idx + 1], rope_cos, rope_sin, pos_ids, kv_indices,
            layer_k, layer_v, pos_id, attn_scale, o_weights[layer_idx:layer_idx + 1],
            mlp_norm_weights[layer_idx], gate_weights[layer_idx:layer_idx + 1],
            up_weights[layer_idx:layer_idx + 1], down_weights[layer_idx:layer_idx + 1],
        )
    final_norm = torch.rms_norm(hidden, [hidden.shape[-1]], lm_head_norm_weight[0], eps.item())
    return final_norm @ lm_head_weight[0].transpose(-1, -2)


def step_11_compiled(hidden, qkv_weights, o_weights, attn_norm_weights, mlp_norm_weights,
                     gate_weights, up_weights, down_weights,
                     k_cache, v_cache, rope_cos, rope_sin,
                     pos_ids, kv_indices, pos_id, attn_scale, eps):
    """Step 11: full layer 0 + layer 1's rms + qkv_rope_append."""
    num_pages_per_layer = k_cache.shape[0] // 2
    layer_k0 = k_cache[0:num_pages_per_layer]
    layer_v0 = v_cache[0:num_pages_per_layer]
    h_norm = torch.ops.megakittens.rms70b(hidden, attn_norm_weights[0], eps)
    q = torch.ops.megakittens.qkv_rope_append70b(
        h_norm, qkv_weights[0:1], rope_cos, rope_sin, pos_ids, kv_indices, layer_k0, layer_v0,
    )
    attn_out = torch.ops.megakittens.attention_decode70b(q, layer_k0, layer_v0, pos_id, attn_scale)
    torch.ops.megakittens.oproj_residual70b(hidden, attn_out, o_weights[0:1])
    mlp_norm = torch.ops.megakittens.rms70b(hidden, mlp_norm_weights[0], eps)
    gate = torch.ops.megakittens.gate_silu70b(mlp_norm, gate_weights[0:1])
    up = torch.ops.megakittens.up_matmul70b(mlp_norm, up_weights[0:1], gate)
    torch.ops.megakittens.oproj_residual70b(hidden, up, down_weights[0:1])
    # Layer 1: rms + qkv_rope_append + attention_decode.
    layer_k1 = k_cache[num_pages_per_layer:2 * num_pages_per_layer]
    layer_v1 = v_cache[num_pages_per_layer:2 * num_pages_per_layer]
    h_norm1 = torch.ops.megakittens.rms70b(hidden, attn_norm_weights[1], eps)
    q1 = torch.ops.megakittens.qkv_rope_append70b(
        h_norm1, qkv_weights[1:2], rope_cos, rope_sin, pos_ids, kv_indices, layer_k1, layer_v1,
    )
    return torch.ops.megakittens.attention_decode70b(q1, layer_k1, layer_v1, pos_id, attn_scale)


def step_11_eager(hidden, qkv_weights, o_weights, attn_norm_weights, mlp_norm_weights,
                  gate_weights, up_weights, down_weights,
                  k_cache, v_cache, rope_cos, rope_sin,
                  pos_ids, kv_indices, pos_id, attn_scale, eps):
    num_pages_per_layer = k_cache.shape[0] // 2
    layer_k0 = k_cache[0:num_pages_per_layer]
    layer_v0 = v_cache[0:num_pages_per_layer]
    hidden = step_8_eager(
        hidden, attn_norm_weights[0], eps,
        qkv_weights[0:1], rope_cos, rope_sin, pos_ids, kv_indices,
        layer_k0, layer_v0, pos_id, attn_scale, o_weights[0:1],
        mlp_norm_weights[0], gate_weights[0:1], up_weights[0:1], down_weights[0:1],
    )
    layer_k1 = k_cache[num_pages_per_layer:2 * num_pages_per_layer]
    layer_v1 = v_cache[num_pages_per_layer:2 * num_pages_per_layer]
    return step_3_eager(
        hidden, attn_norm_weights[1], eps,
        qkv_weights[1:2], rope_cos, rope_sin, pos_ids, kv_indices,
        layer_k1, layer_v1, pos_id, attn_scale,
    )


def _report(label, expected, actual, atol=1e-2, rtol=1e-2):
    diff = (actual.float() - expected.float()).abs()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    bad = (diff > atol + rtol * expected.float().abs()).sum().item()
    total = expected.numel()
    print(f"\n[{label}]:")
    print(f"  shape           : {tuple(actual.shape)}")
    print(f"  max abs diff    : {max_d:.6f}")
    print(f"  mean abs diff   : {mean_d:.6f}")
    print(f"  positions wrong : {bad}/{total}  ({100.0*bad/total:.3f}%)")
    print(f"  expected sample : {expected.flatten()[:6].tolist()}")
    print(f"  actual   sample : {actual.flatten()[:6].tolist()}")


@torch.inference_mode()
def run(step: int, batch_size: int, max_seq_len: int):
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt = "Hello, my name is"
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0].to(device)
    prompt_len = prompt_ids.shape[0]
    print(f"Prompt: {prompt!r} ({prompt_len} tokens)")

    weights, _ = load_hf_weights(batch_size, max_seq_len, device)

    # Real hidden state: the last prompt token embedded across the whole batch.
    last_token = prompt_ids[-1]
    hidden = weights["embed_weight"][last_token].unsqueeze(0).expand(batch_size, HIDDEN_DIM).contiguous().to(torch.bfloat16)
    eps = torch.tensor([RMS_NORM_EPS], dtype=torch.float32, device=device)

    if step == -2:
        # qkv_rope_append ONLY (compiled), fed by eager-rms output, layer 0.
        # Tests qkv in megakernel context but without rms preceding it in the same compile.
        pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        num_pages = batch_size * pages_per_seq
        pos = prompt_len - 1
        pos_ids = torch.full((batch_size,), pos, dtype=torch.int32, device=device)
        kv_indices = _kv_append_indices(batch_size, pages_per_seq, pos, device)

        # Eager rms (so the qkv input is identical to what step 2's chain feeds it).
        attn_norm_w_l0 = weights["attn_norm_weights"][0]
        h_norm = torch.rms_norm(hidden, [hidden.shape[-1]], attn_norm_w_l0, eps.item())

        # Eager run.
        layer_k_e = torch.zeros(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        layer_v_e = torch.zeros_like(layer_k_e)
        expected = qkv_rope_append70b_op_eager_call(
            h_norm.clone(), weights["qkv_weights"][0:1], weights["rope_cos"], weights["rope_sin"],
            pos_ids, kv_indices, layer_k_e, layer_v_e,
        )
        l_k_snap, l_v_snap = layer_k_e.clone(), layer_v_e.clone()

        # Compiled run: only qkv compiled.
        layer_k_a = torch.zeros_like(layer_k_e)
        layer_v_a = torch.zeros_like(layer_k_e)

        def _qkv_only(hidden_norm, qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices, k_cache, v_cache):
            return torch.ops.megakittens.qkv_rope_append70b(
                hidden_norm, qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices, k_cache, v_cache,
            )

        torch._dynamo.reset()
        compiled = megakittens.compile(_qkv_only, use_jit_cache=False, verbose=False, save_schedule=False)
        actual = compiled(
            h_norm.clone(), weights["qkv_weights"][0:1], weights["rope_cos"], weights["rope_sin"],
            pos_ids, kv_indices, layer_k_a, layer_v_a,
        )
        _report("step -2: qkv only (eager rms input)", expected, actual, atol=1e-1, rtol=1e-2)
        _report("step -2: layer 0 k_cache", l_k_snap, layer_k_a, atol=1e-2, rtol=1e-2)
        _report("step -2: layer 0 v_cache", l_v_snap, layer_v_a, atol=1e-2, rtol=1e-2)

    elif step == -1:
        # Just rms+qkv into LAYER 1's slice of a full-NUM_LAYERS cache.
        pos = prompt_len - 1
        pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        num_pages = batch_size * pages_per_seq
        pos_ids = torch.full((batch_size,), pos, dtype=torch.int32, device=device)
        kv_indices = _kv_append_indices(batch_size, pages_per_seq, pos, device)

        # Eager run: write into layer 1's slice of weights["k_cache"]/v_cache.
        weights["k_cache"].zero_()
        weights["v_cache"].zero_()
        l1_k_eager = weights["k_cache"][num_pages:2 * num_pages]
        l1_v_eager = weights["v_cache"][num_pages:2 * num_pages]
        expected = step_2_eager(
            hidden.clone(), weights["attn_norm_weights"][1], eps,
            weights["qkv_weights"][1:2], weights["rope_cos"], weights["rope_sin"], pos_ids, kv_indices,
            l1_k_eager, l1_v_eager,
        )
        # Snapshot eager K/V regions and pre-region (layer 0) for later compare.
        l1_k_snap = l1_k_eager.clone()
        l1_v_snap = l1_v_eager.clone()

        # Compiled run: re-zero, run compiled, compare.
        weights["k_cache"].zero_()
        weights["v_cache"].zero_()

        def _l1_only(hidden, attn_norm_weights, eps, qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices, k_cache, v_cache):
            num_pages_per_layer = k_cache.shape[0] // NUM_LAYERS
            layer_k1 = k_cache[num_pages_per_layer:2 * num_pages_per_layer]
            layer_v1 = v_cache[num_pages_per_layer:2 * num_pages_per_layer]
            h_norm = torch.ops.megakittens.rms70b(hidden, attn_norm_weights[1], eps)
            return torch.ops.megakittens.qkv_rope_append70b(
                h_norm, qkv_weights[1:2], rope_cos, rope_sin, pos_ids, kv_indices, layer_k1, layer_v1,
            )

        torch._dynamo.reset()
        compiled = megakittens.compile(_l1_only, use_jit_cache=False, verbose=False, save_schedule=False)
        actual = compiled(
            hidden.clone(), weights["attn_norm_weights"], eps,
            weights["qkv_weights"], weights["rope_cos"], weights["rope_sin"], pos_ids, kv_indices,
            weights["k_cache"], weights["v_cache"],
        )
        _report("step -1: q output", expected, actual, atol=1e-1, rtol=1e-2)
        l1_k_actual = weights["k_cache"][num_pages:2 * num_pages]
        l1_v_actual = weights["v_cache"][num_pages:2 * num_pages]
        _report("step -1: layer 1 k_cache", l1_k_snap, l1_k_actual, atol=1e-2, rtol=1e-2)
        _report("step -1: layer 1 v_cache", l1_v_snap, l1_v_actual, atol=1e-2, rtol=1e-2)
        # Layers 0 and 2..NUM_LAYERS-1 should be zero on the compiled side.
        for check_l in [0, 2]:
            seg = weights["k_cache"][check_l * num_pages:(check_l + 1) * num_pages]
            print(f"  layer {check_l} k_cache (should be zero): max={seg.abs().max().item():.6f}, nonzero={(seg != 0).sum().item()}/{seg.numel()}")

    elif step == 0:
        # Pure rms-only test on layer 1's norm weight to verify layer_idx routing.
        attn_norm_weight_l1 = weights["attn_norm_weights"][1]
        expected = step_1_eager(hidden, attn_norm_weight_l1, eps)
        torch._dynamo.reset()
        compiled = megakittens.compile(step_1_compiled, use_jit_cache=False, verbose=False, save_schedule=False)
        actual = compiled(hidden.clone(), attn_norm_weight_l1, eps)
        _report("step 0: rms (attn norm, LAYER 1)", expected, actual)

    elif step == 1:
        attn_norm_weight = weights["attn_norm_weights"][0]
        expected = step_1_eager(hidden, attn_norm_weight, eps)
        torch._dynamo.reset()
        compiled = megakittens.compile(step_1_compiled, use_jit_cache=False, verbose=False, save_schedule=False)
        actual = compiled(hidden.clone(), attn_norm_weight, eps)
        _report("step 1: rms (attn norm, layer 0)", expected, actual)

    elif step == 2:
        # KV cache for layer 0 only (one layer's worth of pages).
        pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        num_pages = batch_size * pages_per_seq
        layer_k_e = torch.zeros(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        layer_v_e = torch.zeros_like(layer_k_e)
        layer_k_a = torch.zeros_like(layer_k_e)
        layer_v_a = torch.zeros_like(layer_k_e)

        # Real position state: last prompt token (pos = prompt_len - 1).
        pos = prompt_len - 1
        pos_ids = torch.full((batch_size,), pos, dtype=torch.int32, device=device)
        kv_indices = _kv_append_indices(batch_size, pages_per_seq, pos, device)

        attn_norm_weight = weights["attn_norm_weights"][0]
        qkv_weights = weights["qkv_weights"][0:1]
        rope_cos = weights["rope_cos"]
        rope_sin = weights["rope_sin"]

        expected_q = step_2_eager(
            hidden.clone(), attn_norm_weight, eps,
            qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
            layer_k_e, layer_v_e,
        )
        torch._dynamo.reset()
        compiled = megakittens.compile(step_2_compiled, use_jit_cache=False, verbose=False, save_schedule=True)
        actual_q = compiled(
            hidden.clone(), attn_norm_weight, eps,
            qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
            layer_k_a, layer_v_a,
        )
        _report("step 2: q output", expected_q, actual_q, atol=1e-1, rtol=1e-2)
        _report("step 2: k_cache", layer_k_e, layer_k_a, atol=1e-1, rtol=1e-2)
        _report("step 2: v_cache", layer_v_e, layer_v_a, atol=1e-1, rtol=1e-2)

    elif step == 3:
        pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        num_pages = batch_size * pages_per_seq
        layer_k_e = torch.zeros(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        layer_v_e = torch.zeros_like(layer_k_e)
        layer_k_a = torch.zeros_like(layer_k_e)
        layer_v_a = torch.zeros_like(layer_k_e)

        pos = prompt_len - 1
        pos_ids = torch.full((batch_size,), pos, dtype=torch.int32, device=device)
        kv_indices = _kv_append_indices(batch_size, pages_per_seq, pos, device)
        pos_id = torch.tensor([pos], dtype=torch.int32, device=device)
        attn_scale = torch.tensor([1.0 / math.sqrt(HEAD_DIM)], dtype=torch.float32, device=device)

        attn_norm_weight = weights["attn_norm_weights"][0]
        qkv_weights = weights["qkv_weights"][0:1]
        rope_cos = weights["rope_cos"]
        rope_sin = weights["rope_sin"]

        expected = step_3_eager(
            hidden.clone(), attn_norm_weight, eps,
            qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
            layer_k_e, layer_v_e, pos_id, attn_scale,
        )
        torch._dynamo.reset()
        compiled = megakittens.compile(step_3_compiled, use_jit_cache=False, verbose=False, save_schedule=False)
        actual = compiled(
            hidden.clone(), attn_norm_weight, eps,
            qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
            layer_k_a, layer_v_a, pos_id, attn_scale,
        )
        _report("step 3: attention output", expected, actual, atol=1e-1, rtol=1e-2)

    elif step == 4:
        pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        num_pages = batch_size * pages_per_seq
        layer_k_e = torch.zeros(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        layer_v_e = torch.zeros_like(layer_k_e)
        layer_k_a = torch.zeros_like(layer_k_e)
        layer_v_a = torch.zeros_like(layer_k_e)

        pos = prompt_len - 1
        pos_ids = torch.full((batch_size,), pos, dtype=torch.int32, device=device)
        kv_indices = _kv_append_indices(batch_size, pages_per_seq, pos, device)
        pos_id = torch.tensor([pos], dtype=torch.int32, device=device)
        attn_scale = torch.tensor([1.0 / math.sqrt(HEAD_DIM)], dtype=torch.float32, device=device)

        attn_norm_weight = weights["attn_norm_weights"][0]
        qkv_weights = weights["qkv_weights"][0:1]
        o_weights = weights["o_weights"][0:1]
        rope_cos = weights["rope_cos"]
        rope_sin = weights["rope_sin"]

        hidden_e = hidden.clone()
        hidden_a = hidden.clone()

        expected = step_4_eager(
            hidden_e, attn_norm_weight, eps,
            qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
            layer_k_e, layer_v_e, pos_id, attn_scale, o_weights,
        )
        torch._dynamo.reset()
        compiled = megakittens.compile(step_4_compiled, use_jit_cache=False, verbose=False, save_schedule=False)
        actual = compiled(
            hidden_a, attn_norm_weight, eps,
            qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
            layer_k_a, layer_v_a, pos_id, attn_scale, o_weights,
        )
        _report("step 4: hidden after o_proj+residual", expected, actual, atol=1e-1, rtol=1e-2)

    elif step == 5:
        pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        num_pages = batch_size * pages_per_seq
        layer_k_e = torch.zeros(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        layer_v_e = torch.zeros_like(layer_k_e)
        layer_k_a = torch.zeros_like(layer_k_e)
        layer_v_a = torch.zeros_like(layer_k_e)

        pos = prompt_len - 1
        pos_ids = torch.full((batch_size,), pos, dtype=torch.int32, device=device)
        kv_indices = _kv_append_indices(batch_size, pages_per_seq, pos, device)
        pos_id = torch.tensor([pos], dtype=torch.int32, device=device)
        attn_scale = torch.tensor([1.0 / math.sqrt(HEAD_DIM)], dtype=torch.float32, device=device)

        attn_norm_weight = weights["attn_norm_weights"][0]
        mlp_norm_weight = weights["mlp_norm_weights"][0]
        qkv_weights = weights["qkv_weights"][0:1]
        o_weights = weights["o_weights"][0:1]
        rope_cos = weights["rope_cos"]
        rope_sin = weights["rope_sin"]

        hidden_e = hidden.clone()
        hidden_a = hidden.clone()

        expected = step_5_eager(
            hidden_e, attn_norm_weight, eps,
            qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
            layer_k_e, layer_v_e, pos_id, attn_scale, o_weights, mlp_norm_weight,
        )
        torch._dynamo.reset()
        compiled = megakittens.compile(step_5_compiled, use_jit_cache=False, verbose=False, save_schedule=False)
        actual = compiled(
            hidden_a, attn_norm_weight, eps,
            qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
            layer_k_a, layer_v_a, pos_id, attn_scale, o_weights, mlp_norm_weight,
        )
        _report("step 5: mlp_norm output", expected, actual, atol=1e-1, rtol=1e-2)

    elif step == 6:
        pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        num_pages = batch_size * pages_per_seq
        layer_k_e = torch.zeros(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        layer_v_e = torch.zeros_like(layer_k_e)
        layer_k_a = torch.zeros_like(layer_k_e)
        layer_v_a = torch.zeros_like(layer_k_e)

        pos = prompt_len - 1
        pos_ids = torch.full((batch_size,), pos, dtype=torch.int32, device=device)
        kv_indices = _kv_append_indices(batch_size, pages_per_seq, pos, device)
        pos_id = torch.tensor([pos], dtype=torch.int32, device=device)
        attn_scale = torch.tensor([1.0 / math.sqrt(HEAD_DIM)], dtype=torch.float32, device=device)

        attn_norm_weight = weights["attn_norm_weights"][0]
        mlp_norm_weight = weights["mlp_norm_weights"][0]
        qkv_weights = weights["qkv_weights"][0:1]
        o_weights = weights["o_weights"][0:1]
        gate_weights = weights["gate_weights"][0:1]
        rope_cos = weights["rope_cos"]
        rope_sin = weights["rope_sin"]

        hidden_e = hidden.clone()
        hidden_a = hidden.clone()

        expected = step_6_eager(
            hidden_e, attn_norm_weight, eps,
            qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
            layer_k_e, layer_v_e, pos_id, attn_scale, o_weights,
            mlp_norm_weight, gate_weights,
        )
        torch._dynamo.reset()
        compiled = megakittens.compile(step_6_compiled, use_jit_cache=False, verbose=False, save_schedule=False)
        actual = compiled(
            hidden_a, attn_norm_weight, eps,
            qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
            layer_k_a, layer_v_a, pos_id, attn_scale, o_weights,
            mlp_norm_weight, gate_weights,
        )
        _report("step 6: gate_silu output", expected, actual, atol=1e-1, rtol=1e-2)

    elif step == 7:
        pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        num_pages = batch_size * pages_per_seq
        layer_k_e = torch.zeros(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        layer_v_e = torch.zeros_like(layer_k_e)
        layer_k_a = torch.zeros_like(layer_k_e)
        layer_v_a = torch.zeros_like(layer_k_e)

        pos = prompt_len - 1
        pos_ids = torch.full((batch_size,), pos, dtype=torch.int32, device=device)
        kv_indices = _kv_append_indices(batch_size, pages_per_seq, pos, device)
        pos_id = torch.tensor([pos], dtype=torch.int32, device=device)
        attn_scale = torch.tensor([1.0 / math.sqrt(HEAD_DIM)], dtype=torch.float32, device=device)

        attn_norm_weight = weights["attn_norm_weights"][0]
        mlp_norm_weight = weights["mlp_norm_weights"][0]
        qkv_weights = weights["qkv_weights"][0:1]
        o_weights = weights["o_weights"][0:1]
        gate_weights = weights["gate_weights"][0:1]
        up_weights = weights["up_weights"][0:1]
        rope_cos = weights["rope_cos"]
        rope_sin = weights["rope_sin"]

        hidden_e = hidden.clone()
        hidden_a = hidden.clone()

        expected = step_7_eager(
            hidden_e, attn_norm_weight, eps,
            qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
            layer_k_e, layer_v_e, pos_id, attn_scale, o_weights,
            mlp_norm_weight, gate_weights, up_weights,
        )
        torch._dynamo.reset()
        compiled = megakittens.compile(step_7_compiled, use_jit_cache=False, verbose=False, save_schedule=False)
        actual = compiled(
            hidden_a, attn_norm_weight, eps,
            qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
            layer_k_a, layer_v_a, pos_id, attn_scale, o_weights,
            mlp_norm_weight, gate_weights, up_weights,
        )
        _report("step 7: up_matmul output", expected, actual, atol=1e-1, rtol=1e-2)

    elif step == 8:
        pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        num_pages = batch_size * pages_per_seq
        layer_k_e = torch.zeros(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        layer_v_e = torch.zeros_like(layer_k_e)
        layer_k_a = torch.zeros_like(layer_k_e)
        layer_v_a = torch.zeros_like(layer_k_e)

        pos = prompt_len - 1
        pos_ids = torch.full((batch_size,), pos, dtype=torch.int32, device=device)
        kv_indices = _kv_append_indices(batch_size, pages_per_seq, pos, device)
        pos_id = torch.tensor([pos], dtype=torch.int32, device=device)
        attn_scale = torch.tensor([1.0 / math.sqrt(HEAD_DIM)], dtype=torch.float32, device=device)

        attn_norm_weight = weights["attn_norm_weights"][0]
        mlp_norm_weight = weights["mlp_norm_weights"][0]
        qkv_weights = weights["qkv_weights"][0:1]
        o_weights = weights["o_weights"][0:1]
        gate_weights = weights["gate_weights"][0:1]
        up_weights = weights["up_weights"][0:1]
        down_weights = weights["down_weights"][0:1]
        rope_cos = weights["rope_cos"]
        rope_sin = weights["rope_sin"]

        hidden_e = hidden.clone()
        hidden_a = hidden.clone()

        expected = step_8_eager(
            hidden_e, attn_norm_weight, eps,
            qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
            layer_k_e, layer_v_e, pos_id, attn_scale, o_weights,
            mlp_norm_weight, gate_weights, up_weights, down_weights,
        )
        torch._dynamo.reset()
        compiled = megakittens.compile(step_8_compiled, use_jit_cache=False, verbose=False, save_schedule=False)
        actual = compiled(
            hidden_a, attn_norm_weight, eps,
            qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
            layer_k_a, layer_v_a, pos_id, attn_scale, o_weights,
            mlp_norm_weight, gate_weights, up_weights, down_weights,
        )
        _report("step 8: hidden after full layer 0", expected, actual, atol=1e-1, rtol=1e-2)

    elif step == 9:
        pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        num_pages = batch_size * pages_per_seq
        layer_k_e = torch.zeros(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        layer_v_e = torch.zeros_like(layer_k_e)
        layer_k_a = torch.zeros_like(layer_k_e)
        layer_v_a = torch.zeros_like(layer_k_e)

        pos = prompt_len - 1
        pos_ids = torch.full((batch_size,), pos, dtype=torch.int32, device=device)
        kv_indices = _kv_append_indices(batch_size, pages_per_seq, pos, device)
        pos_id = torch.tensor([pos], dtype=torch.int32, device=device)
        attn_scale = torch.tensor([1.0 / math.sqrt(HEAD_DIM)], dtype=torch.float32, device=device)

        attn_norm_weight = weights["attn_norm_weights"][0]
        mlp_norm_weight = weights["mlp_norm_weights"][0]
        qkv_weights = weights["qkv_weights"][0:1]
        o_weights = weights["o_weights"][0:1]
        gate_weights = weights["gate_weights"][0:1]
        up_weights = weights["up_weights"][0:1]
        down_weights = weights["down_weights"][0:1]
        lm_head_norm_weight = weights["lm_head_norm_weight"]
        lm_head_weight = weights["lm_head_weight"]
        rope_cos = weights["rope_cos"]
        rope_sin = weights["rope_sin"]

        hidden_e = hidden.clone()
        hidden_a = hidden.clone()

        expected = step_9_eager(
            hidden_e, attn_norm_weight, eps,
            qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
            layer_k_e, layer_v_e, pos_id, attn_scale, o_weights,
            mlp_norm_weight, gate_weights, up_weights, down_weights,
            lm_head_norm_weight, lm_head_weight,
        )
        torch._dynamo.reset()
        compiled = megakittens.compile(step_9_compiled, use_jit_cache=False, verbose=False, save_schedule=False)
        actual = compiled(
            hidden_a, attn_norm_weight, eps,
            qkv_weights, rope_cos, rope_sin, pos_ids, kv_indices,
            layer_k_a, layer_v_a, pos_id, attn_scale, o_weights,
            mlp_norm_weight, gate_weights, up_weights, down_weights,
            lm_head_norm_weight, lm_head_weight,
        )
        _report("step 9: logits (layer 0 + final rms + lm_head)", expected, actual, atol=1e-1, rtol=1e-2)

        # Argmax comparison too: this is what decoding sees
        expected_argmax = expected.argmax(dim=-1)
        actual_argmax = actual.argmax(dim=-1)
        agree = (expected_argmax == actual_argmax).sum().item()
        print(f"  argmax agreement: {agree}/{batch_size}")
        print(f"  expected argmax sample: {expected_argmax[:8].tolist()}")
        print(f"  actual   argmax sample: {actual_argmax[:8].tolist()}")

    elif step == 10:
        # Run on full NUM_LAYERS k_cache (shared across eager + compiled to fit memory).
        num_layers = 2
        pos = prompt_len - 1
        pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        pos_ids = torch.full((batch_size,), pos, dtype=torch.int32, device=device)
        kv_indices = _kv_append_indices(batch_size, pages_per_seq, pos, device)
        pos_id = torch.tensor([pos], dtype=torch.int32, device=device)
        attn_scale = torch.tensor([1.0 / math.sqrt(HEAD_DIM)], dtype=torch.float32, device=device)

        # Eager
        weights["k_cache"].zero_()
        weights["v_cache"].zero_()
        hidden_e = hidden.clone()
        expected = step_10_eager(
            hidden_e, weights["qkv_weights"], weights["o_weights"],
            weights["attn_norm_weights"], weights["mlp_norm_weights"],
            weights["gate_weights"], weights["up_weights"], weights["down_weights"],
            weights["lm_head_norm_weight"], weights["lm_head_weight"],
            weights["k_cache"], weights["v_cache"], weights["rope_cos"], weights["rope_sin"],
            pos_ids, kv_indices, pos_id, attn_scale, eps, num_layers,
        )
        expected = expected.clone()  # snapshot before re-running

        # Compiled
        weights["k_cache"].zero_()
        weights["v_cache"].zero_()
        hidden_a = hidden.clone()
        torch._dynamo.reset()
        compiled = megakittens.compile(step_10_compiled, use_jit_cache=False, verbose=False, save_schedule=False)
        actual = compiled(
            hidden_a, weights["qkv_weights"], weights["o_weights"],
            weights["attn_norm_weights"], weights["mlp_norm_weights"],
            weights["gate_weights"], weights["up_weights"], weights["down_weights"],
            weights["lm_head_norm_weight"], weights["lm_head_weight"],
            weights["k_cache"], weights["v_cache"], weights["rope_cos"], weights["rope_sin"],
            pos_ids, kv_indices, pos_id, attn_scale, eps, num_layers,
        )
        _report(f"step 10: logits ({num_layers} layers + rms + lm_head)", expected, actual, atol=1e-1, rtol=1e-2)
        expected_argmax = expected.argmax(dim=-1)
        actual_argmax = actual.argmax(dim=-1)
        agree = (expected_argmax == actual_argmax).sum().item()
        print(f"  argmax agreement: {agree}/{batch_size}")
        print(f"  expected argmax sample: {expected_argmax[:8].tolist()}")
        print(f"  actual   argmax sample: {actual_argmax[:8].tolist()}")

    elif step == 11:
        num_layers = 2
        pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        num_pages = batch_size * pages_per_seq
        k_cache_e = torch.zeros(num_layers * num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        v_cache_e = torch.zeros_like(k_cache_e)
        k_cache_a = torch.zeros_like(k_cache_e)
        v_cache_a = torch.zeros_like(k_cache_e)

        pos = prompt_len - 1
        pos_ids = torch.full((batch_size,), pos, dtype=torch.int32, device=device)
        kv_indices = _kv_append_indices(batch_size, pages_per_seq, pos, device)
        pos_id = torch.tensor([pos], dtype=torch.int32, device=device)
        attn_scale = torch.tensor([1.0 / math.sqrt(HEAD_DIM)], dtype=torch.float32, device=device)

        hidden_e = hidden.clone()
        hidden_a = hidden.clone()

        expected = step_11_eager(
            hidden_e, weights["qkv_weights"], weights["o_weights"],
            weights["attn_norm_weights"], weights["mlp_norm_weights"],
            weights["gate_weights"], weights["up_weights"], weights["down_weights"],
            k_cache_e, v_cache_e, weights["rope_cos"], weights["rope_sin"],
            pos_ids, kv_indices, pos_id, attn_scale, eps,
        )
        torch._dynamo.reset()
        compiled = megakittens.compile(step_11_compiled, use_jit_cache=False, verbose=False, save_schedule=False)
        actual = compiled(
            hidden_a, weights["qkv_weights"], weights["o_weights"],
            weights["attn_norm_weights"], weights["mlp_norm_weights"],
            weights["gate_weights"], weights["up_weights"], weights["down_weights"],
            k_cache_a, v_cache_a, weights["rope_cos"], weights["rope_sin"],
            pos_ids, kv_indices, pos_id, attn_scale, eps,
        )
        _report("step 11: q output (layer 0 full + layer 1 rms+qkv)", expected, actual, atol=1e-1, rtol=1e-2)
        # Also verify layer 1's K and V cache slices match.
        l1_k_e = k_cache_e[num_pages:2 * num_pages]
        l1_k_a = k_cache_a[num_pages:2 * num_pages]
        l1_v_e = v_cache_e[num_pages:2 * num_pages]
        l1_v_a = v_cache_a[num_pages:2 * num_pages]
        _report("step 11: layer 1 k_cache", l1_k_e, l1_k_a, atol=1e-2, rtol=1e-2)
        _report("step 11: layer 1 v_cache", l1_v_e, l1_v_a, atol=1e-2, rtol=1e-2)
        # Also check layer 0's K cache was untouched by layer 1.
        l0_k_e = k_cache_e[0:num_pages]
        l0_k_a = k_cache_a[0:num_pages]
        _report("step 11: layer 0 k_cache (should match)", l0_k_e, l0_k_a, atol=1e-2, rtol=1e-2)

    else:
        raise NotImplementedError(f"step {step} not yet implemented")


if __name__ == "__main__":
    initialize_cuda_context()
    torch._dynamo.reset()
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    args = parser.parse_args()
    run(step=args.step, batch_size=args.batch_size, max_seq_len=args.max_seq_len)
