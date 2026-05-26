"""Llama-3.2-1B decode using megakittens.compile"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

import megakittens
from megakittens.jit.cuda_utils import initialize_cuda_context


NUM_LAYERS = 16
HIDDEN_DIM = 2048
INTERMEDIATE_DIM = 8192
NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64
VOCAB_SIZE = 128256
RMS_NORM_EPS = 1e-5
MAX_SEQ_LEN = 4096
Q_DIM = NUM_ATTENTION_HEADS * HEAD_DIM
K_DIM = NUM_KV_HEADS * HEAD_DIM
ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)


def _apply_rope(x, cos, sin):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return (x * cos + rotated * sin).to(x.dtype)


def _rmsnorm(x, weight, eps):
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    normed = x_float * torch.rsqrt(variance + eps)
    return (weight.float() * normed).to(x.dtype)


def _interleave_indices(num_heads, head_dim):
    half = head_dim // 2
    indices = []
    for h in range(num_heads):
        offset = h * head_dim
        for i in range(half):
            indices.append(offset + i)
            indices.append(offset + half + i)
    return torch.tensor(indices)


def _stack_weights(hf_model):
    config = hf_model.config
    model = hf_model.model
    q_indices = _interleave_indices(config.num_attention_heads, config.head_dim)
    k_indices = _interleave_indices(config.num_key_value_heads, config.head_dim)

    qkv_weights, o_weights = [], []
    attn_norm_weights, mlp_norm_weights = [], []
    up_weights, gate_weights, down_weights = [], [], []

    for layer in model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        qkv_weights.append(torch.cat([attn.q_proj.weight[q_indices],
                                       attn.k_proj.weight[k_indices],
                                       attn.v_proj.weight], dim=0))
        o_weights.append(attn.o_proj.weight)
        attn_norm_weights.append(layer.input_layernorm.weight)
        mlp_norm_weights.append(layer.post_attention_layernorm.weight)
        up_weights.append(mlp.up_proj.weight)
        gate_weights.append(mlp.gate_proj.weight)
        down_weights.append(mlp.down_proj.weight)

    return {
        "qkv_weights": torch.stack(qkv_weights),
        "o_weights": torch.stack(o_weights),
        "attn_norm_weights": torch.stack(attn_norm_weights),
        "mlp_norm_weights": torch.stack(mlp_norm_weights),
        "up_weights": torch.stack(up_weights),
        "gate_weights": torch.stack(gate_weights),
        "down_weights": torch.stack(down_weights),
        "lm_head_norm_weight": model.norm.weight,
        "lm_head_weight": hf_model.lm_head.weight,
        "embed_weight": model.embed_tokens.weight,
    }


def _make_rope_table(config, max_seq_len, device):
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    rope = LlamaRotaryEmbedding(config=config)
    positions = torch.arange(max_seq_len).unsqueeze(0)
    dummy = torch.empty(0, config.hidden_size, dtype=torch.float32)
    cos_hf, sin_hf = rope(dummy, positions)
    cos_hf = cos_hf.squeeze(0).to(device)
    sin_hf = sin_hf.squeeze(0).to(device)
    one_head_indices = _interleave_indices(1, config.head_dim)
    return cos_hf[..., one_head_indices], sin_hf[..., one_head_indices]


def _prefill_kv_cache(token_ids, weights, k_cache, v_cache, rope_cos, rope_sin):
    for pos in range(len(token_ids)):
        x = weights["embed_weight"][token_ids[pos]]
        cos = rope_cos[pos]
        sin = rope_sin[pos]
        seq_len = pos + 1

        for layer_idx in range(NUM_LAYERS):
            normed = _rmsnorm(x, weights["attn_norm_weights"][layer_idx], RMS_NORM_EPS)
            qkv = weights["qkv_weights"][layer_idx] @ normed
            q = qkv[:Q_DIM].view(NUM_ATTENTION_HEADS, HEAD_DIM)
            k = qkv[Q_DIM:Q_DIM + K_DIM].view(NUM_KV_HEADS, HEAD_DIM)
            v = qkv[Q_DIM + K_DIM:].view(NUM_KV_HEADS, HEAD_DIM)

            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
            k_cache[layer_idx, pos] = k
            v_cache[layer_idx, pos] = v

            attn_out = torch.zeros(NUM_ATTENTION_HEADS, HEAD_DIM, device=x.device, dtype=x.dtype)
            for kv_head in range(NUM_KV_HEADS):
                k_cached = k_cache[layer_idx, :seq_len, kv_head]
                v_cached = v_cache[layer_idx, :seq_len, kv_head]
                gqa_size = NUM_ATTENTION_HEADS // NUM_KV_HEADS
                for q_head in range(kv_head * gqa_size, (kv_head + 1) * gqa_size):
                    scores = (q[q_head] @ k_cached.T) * ATTN_SCALE
                    if seq_len > 1:
                        mask = torch.full((seq_len,), float("-inf"), device=x.device)
                        mask[:pos + 1] = 0.0
                        scores = scores + mask
                    w = F.softmax(scores.float(), dim=-1).to(x.dtype)
                    attn_out[q_head] = w @ v_cached

            x = x + weights["o_weights"][layer_idx] @ attn_out.reshape(HIDDEN_DIM)

            normed_mlp = _rmsnorm(x, weights["mlp_norm_weights"][layer_idx], RMS_NORM_EPS)
            gate = weights["gate_weights"][layer_idx] @ normed_mlp
            up = weights["up_weights"][layer_idx] @ normed_mlp
            x = x + weights["down_weights"][layer_idx] @ (F.silu(gate) * up)

    return x


def decode(
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
    for i in range(NUM_LAYERS):
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

        torch.ops.megakittens.mat_vec_adds(hidden_states, attn_out, o_weights[i:i+1])

        silu_out = torch.ops.megakittens.rms_upgate_silu(
            hidden_states,
            mlp_norm_weights[i:i+1],
            up_weights[i:i+1],
            gate_weights[i:i+1],
            rms_norm_eps,
        )

        torch.ops.megakittens.mat_vec_adds(
            hidden_states, silu_out, down_weights[i:i+1],
        )

    logits = torch.ops.megakittens.rms_lm_head(
        hidden_states, lm_head_norm_weight, lm_head_weight, rms_norm_eps,
    )
    torch.ops.megakittens.pos_id_increment(pos_id)
    return logits


@torch.inference_mode()
def benchmark_tok_per_sec(prompt="Hello, my name is", max_new_tokens=200, num_samples=5, warmup=5):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    D = "cuda"

    print("Loading Llama-3.2-1B weights from HuggingFace...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", dtype=torch.bfloat16, device_map=D,
    )
    weights = _stack_weights(hf_model)
    rope_cos, rope_sin = _make_rope_table(hf_model.config, MAX_SEQ_LEN, D)
    embed_weight = weights["embed_weight"]

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
    prompt_len = input_ids.shape[0]
    print(f"Prompt: {prompt!r}, {prompt_len} tokens")

    del hf_model

    k_cache = torch.zeros(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    v_cache = torch.zeros(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    print("Prefilling KV cache...")
    with torch.inference_mode():
        last_hidden = _prefill_kv_cache(input_ids, weights, k_cache, v_cache, rope_cos, rope_sin)
    prefill_logits = weights["lm_head_weight"] @ _rmsnorm(last_hidden, weights["lm_head_norm_weight"], RMS_NORM_EPS)
    first_token = torch.argmax(prefill_logits)
    k_cache_snapshot = k_cache.clone()
    v_cache_snapshot = v_cache.clone()

    hidden_states = embed_weight[first_token].clone()
    pos_id_tensor = torch.tensor([prompt_len], dtype=torch.int32, device=D)
    attn_scale_tensor = torch.tensor([ATTN_SCALE], dtype=torch.float32, device=D)
    rms_norm_eps_tensor = torch.tensor([RMS_NORM_EPS], dtype=torch.float32, device=D)

    decode_args = (
        hidden_states,
        weights["qkv_weights"], weights["o_weights"],
        weights["attn_norm_weights"], weights["mlp_norm_weights"],
        weights["up_weights"], weights["gate_weights"], weights["down_weights"],
        weights["lm_head_norm_weight"], weights["lm_head_weight"],
        k_cache, v_cache, rope_cos, rope_sin,
        pos_id_tensor, attn_scale_tensor, rms_norm_eps_tensor,
    )

    compiled = megakittens.compile(decode, use_jit_cache=False, verbose=False, save_schedule=False, cluster_size=1)

    weight_tensors = [
        weights["qkv_weights"], weights["o_weights"],
        weights["attn_norm_weights"], weights["mlp_norm_weights"],
        weights["up_weights"], weights["gate_weights"], weights["down_weights"],
        weights["lm_head_norm_weight"], weights["lm_head_weight"],
    ]
    model_size = sum(t.nelement() * t.element_size() for t in weight_tensors)
    params = sum(t.nelement() for t in weight_tensors)

    embedding = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_DIM, device=D, dtype=torch.bfloat16)
    embedding.weight.data.copy_(embed_weight)
    num_decode_tokens = max_new_tokens - 1
    output_tokens = torch.zeros(max_new_tokens, dtype=torch.long, device=D)

    def _decode_step(input_token):
        hidden_states.copy_(embedding(input_token))
        logits = compiled(*decode_args)
        return torch.argmax(logits, dim=-1)

    pos_id_tensor.fill_(prompt_len)
    compiled(*decode_args)
    output_tokens[0] = first_token
    print(f"Warming up ({warmup} runs)...")
    for _ in range(warmup):
        k_cache.copy_(k_cache_snapshot)
        v_cache.copy_(v_cache_snapshot)
        pos_id_tensor.fill_(prompt_len)
        token = first_token
        for i in range(num_decode_tokens):
            token = _decode_step(token)
            output_tokens[i + 1] = token
    torch.cuda.synchronize()
    print("Warmup done.")

    decode_tokens_per_sec_list = []
    for sample in range(num_samples):
        k_cache.copy_(k_cache_snapshot)
        v_cache.copy_(v_cache_snapshot)
        pos_id_tensor.fill_(prompt_len)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        token = first_token
        for i in range(num_decode_tokens):
            token = _decode_step(token)
            output_tokens[i + 1] = token
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        decode_time = t1 - t0
        decode_tok_sec = num_decode_tokens / decode_time
        decode_tokens_per_sec_list.append(decode_tok_sec)
        bandwidth_gbs = model_size * decode_tok_sec / 1e9
        flops_tfs = params * decode_tok_sec * 2 / 1e12
        print(f"Time for inference {sample + 1}: {decode_time:.02f} sec total, {decode_tok_sec:.02f} tokens/sec")
        print(f"Bandwidth achieved: {bandwidth_gbs:.02f} GB/s")
        print(f"FLOPS achieved: {flops_tfs:.02f} TF/s")
        print()

    all_ids = torch.cat([input_ids.to(D), output_tokens[:max_new_tokens]])
    print(tokenizer.decode(all_ids.tolist()))
    print()

    print("==========")
    print(f"Prompt Length: {prompt_len}")
    print(f"Generated tokens: {max_new_tokens}")
    print(f"Average tokens/sec (decode only): {torch.mean(torch.tensor(decode_tokens_per_sec_list)).item():.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == "__main__":
    import argparse
    initialize_cuda_context()
    torch._dynamo.reset()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    benchmark_tok_per_sec(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        warmup=args.warmup,
    )
