"""Single-GPU batched Llama-3.3-70B decode using megakittens.compile."""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import megakittens
from megakittens.jit.cuda_utils import initialize_cuda_context


MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

NUM_LAYERS = 80
HIDDEN_DIM = 8192
HEAD_DIM = 128
NUM_Q_HEADS = 64
NUM_KV_HEADS = 8
GQA_RATIO = NUM_Q_HEADS // NUM_KV_HEADS
Q_DIM = NUM_Q_HEADS * HEAD_DIM
KV_DIM = NUM_KV_HEADS * HEAD_DIM
QKV_DIM = Q_DIM + 2 * KV_DIM
INTERMEDIATE_DIM = 28672
VOCAB_SIZE = 128256
PAGE_SIZE = 128
RMS_NORM_EPS = 1e-5
ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)


def _interleave_indices(num_heads: int, head_dim: int, device=None) -> torch.Tensor:
    half = head_dim // 2
    indices = []
    for head in range(num_heads):
        offset = head * head_dim
        for i in range(half):
            indices.append(offset + i)
            indices.append(offset + half + i)
    return torch.tensor(indices, dtype=torch.long, device=device)


def _make_rope_table(config, max_seq_len: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    rope = LlamaRotaryEmbedding(config=config)
    positions = torch.arange(max_seq_len).unsqueeze(0)
    dummy = torch.empty(0, config.hidden_size, dtype=torch.float32)
    cos_hf, sin_hf = rope(dummy, positions)
    cos_hf = cos_hf.squeeze(0).to(device)
    sin_hf = sin_hf.squeeze(0).to(device)
    one_head_indices = _interleave_indices(1, HEAD_DIM, device=device)
    return cos_hf[..., one_head_indices].contiguous(), sin_hf[..., one_head_indices].contiguous()


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
        hidden_norm = torch.ops.megakittens.rms70b(
            hidden, attn_norm_weights[layer_idx], rms_norm_eps,
        )
        layer_page_start = layer_idx * num_pages
        layer_page_stop = layer_page_start + num_pages
        layer_k = k_cache[layer_page_start:layer_page_stop]
        layer_v = v_cache[layer_page_start:layer_page_stop]
        q = torch.ops.megakittens.qkv_rope_append70b(
            hidden_norm,
            qkv_weights[layer_idx:layer_idx + 1],
            rope_cos,
            rope_sin,
            pos_id,
            kv_append_indices,
            layer_k,
            layer_v,
        )

        attn_out = torch.ops.megakittens.attention_decode70b(
            q,
            layer_k,
            layer_v,
            pos_id,
            attn_scale,
        )

        torch.ops.megakittens.oproj_residual_half_tmem70b(
            hidden, attn_out, o_weights[layer_idx:layer_idx + 1],
        )

        mlp_norm = torch.ops.megakittens.rms70b(
            hidden, mlp_norm_weights[layer_idx], rms_norm_eps,
        )
        gate = torch.ops.megakittens.gate_silu70b(
            mlp_norm, gate_weights[layer_idx:layer_idx + 1],
        )
        up = torch.ops.megakittens.up_matmul70b(
            mlp_norm, up_weights[layer_idx:layer_idx + 1], gate,
        )

        torch.ops.megakittens.oproj_residual_half_tmem70b(
            hidden, up, down_weights[layer_idx:layer_idx + 1],
        )

    logits_hidden = torch.ops.megakittens.rms70b(
        hidden, lm_head_norm_weight[0], rms_norm_eps,
    )
    logits = torch.ops.megakittens.lm_head70b(logits_hidden, lm_head_weight)
    torch.ops.megakittens.pos_id_increment(pos_id)
    return logits


def load_hf_weights(
    batch_size: int,
    max_seq_len: int,
    device: str,
    num_layers: int,
) -> tuple[dict[str, torch.Tensor], int]:
    print(f"Fetching {MODEL_ID}...")
    config = AutoConfig.from_pretrained(MODEL_ID)
    repo_dir = snapshot_download(MODEL_ID, allow_patterns=["*.safetensors", "*.json"])

    pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = batch_size * pages_per_seq

    weights = {
        "qkv_weights": torch.empty(num_layers, QKV_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device),
        "o_weights": torch.empty(num_layers, HIDDEN_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device),
        "attn_norm_weights": torch.empty(num_layers, HIDDEN_DIM, dtype=torch.bfloat16, device=device),
        "mlp_norm_weights": torch.empty(num_layers, HIDDEN_DIM, dtype=torch.bfloat16, device=device),
        "gate_weights": torch.empty(num_layers, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device),
        "up_weights": torch.empty(num_layers, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device),
        "down_weights": torch.empty(num_layers, HIDDEN_DIM, INTERMEDIATE_DIM, dtype=torch.bfloat16, device=device),
        "lm_head_norm_weight": torch.empty(1, HIDDEN_DIM, dtype=torch.bfloat16, device=device),
        "lm_head_weight": torch.empty(1, VOCAB_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=device),
        "embed_weight": torch.empty(VOCAB_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=device),
        "k_cache": torch.zeros(num_layers * num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device),
        "v_cache": torch.zeros(num_layers * num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device),
    }

    q_idx = _interleave_indices(NUM_Q_HEADS, HEAD_DIM, device=device)
    k_idx = _interleave_indices(NUM_KV_HEADS, HEAD_DIM, device=device)

    with open(os.path.join(repo_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        shards.setdefault(shard, []).append(name)

    print(f"Streaming {len(shards)} safetensors shards into GPU buffers...")
    for shard_file, names in shards.items():
        with safe_open(os.path.join(repo_dir, shard_file), framework="pt", device=device) as f:
            for name in names:
                t = f.get_tensor(name)
                if name == "model.embed_tokens.weight":
                    weights["embed_weight"].copy_(t)
                elif name == "model.norm.weight":
                    weights["lm_head_norm_weight"][0].copy_(t)
                elif name == "lm_head.weight":
                    weights["lm_head_weight"][0].copy_(t)
                elif name.startswith("model.layers."):
                    parts = name.split(".")
                    layer = int(parts[2])
                    if layer >= num_layers:
                        continue
                    suffix = ".".join(parts[3:])
                    if suffix == "input_layernorm.weight":
                        weights["attn_norm_weights"][layer].copy_(t)
                    elif suffix == "post_attention_layernorm.weight":
                        weights["mlp_norm_weights"][layer].copy_(t)
                    elif suffix == "self_attn.q_proj.weight":
                        weights["qkv_weights"][layer, :Q_DIM].copy_(t[q_idx])
                    elif suffix == "self_attn.k_proj.weight":
                        weights["qkv_weights"][layer, Q_DIM:Q_DIM + KV_DIM].copy_(t[k_idx])
                    elif suffix == "self_attn.v_proj.weight":
                        weights["qkv_weights"][layer, Q_DIM + KV_DIM:].copy_(t)
                    elif suffix == "self_attn.o_proj.weight":
                        weights["o_weights"][layer].copy_(t)
                    elif suffix == "mlp.gate_proj.weight":
                        weights["gate_weights"][layer].copy_(t)
                    elif suffix == "mlp.up_proj.weight":
                        weights["up_weights"][layer].copy_(t)
                    elif suffix == "mlp.down_proj.weight":
                        weights["down_weights"][layer].copy_(t)

    weights["rope_cos"], weights["rope_sin"] = _make_rope_table(config, max_seq_len, device)
    return weights, pages_per_seq


@torch.inference_mode()
def benchmark_tok_per_sec(
    decode_fn,
    *,
    num_layers: int,
    prompt: str = "Hello, my name is",
    batch_size: int = 512,
    max_seq_len: int = 128,
    max_new_tokens: int = 64,
    num_samples: int = 3,
    warmup: int = 2,
):
    if batch_size % 512 != 0:
        raise ValueError("Current 70B matmul itypes require batch_size divisible by 512.")

    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0].to(device)
    prompt_len = prompt_ids.shape[0]
    print(f"Prompt: {prompt!r} ({prompt_len} tokens)")

    needed_seq_len = ((prompt_len + max_new_tokens + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    if needed_seq_len > max_seq_len:
        print(f"Bumping max_seq_len: {max_seq_len} -> {needed_seq_len} to fit prompt + max_new_tokens")
        max_seq_len = needed_seq_len

    weights, pages_per_seq = load_hf_weights(batch_size, max_seq_len, device, num_layers)
    num_pages = batch_size * pages_per_seq

    hidden = torch.empty(batch_size, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    kv_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
    kv_index_base = (
        torch.arange(batch_size, dtype=torch.int32, device=device)
        * pages_per_seq * PAGE_SIZE
    )
    pos_id = torch.zeros(1, dtype=torch.int32, device=device)
    attn_scale = torch.tensor([ATTN_SCALE], dtype=torch.float32, device=device)
    rms_norm_eps = torch.tensor([RMS_NORM_EPS], dtype=torch.float32, device=device)

    decode_args = (
        hidden,
        weights["qkv_weights"], weights["o_weights"],
        weights["attn_norm_weights"], weights["mlp_norm_weights"],
        weights["gate_weights"], weights["up_weights"], weights["down_weights"],
        weights["lm_head_norm_weight"], weights["lm_head_weight"],
        weights["k_cache"], weights["v_cache"], weights["rope_cos"], weights["rope_sin"],
        kv_indices, pos_id, attn_scale, rms_norm_eps,
    )

    weight_tensors = [
        weights["qkv_weights"], weights["o_weights"], weights["attn_norm_weights"],
        weights["mlp_norm_weights"], weights["gate_weights"], weights["up_weights"],
        weights["down_weights"], weights["lm_head_norm_weight"], weights["lm_head_weight"],
    ]
    model_size = sum(t.nelement() * t.element_size() for t in weight_tensors)
    params = sum(t.nelement() for t in weight_tensors)

    prompt_tokens = prompt_ids.unsqueeze(1).expand(prompt_len, batch_size).contiguous()
    num_decode_tokens = max_new_tokens - 1
    output_tokens = torch.empty(max_new_tokens, dtype=torch.long, device=device)

    def _decode_step(pos: int, input_tokens: torch.Tensor) -> torch.Tensor:
        hidden.copy_(weights["embed_weight"][input_tokens])
        torch.add(kv_index_base, pos, out=kv_indices)
        logits = decode_fn(*decode_args)
        return torch.argmax(logits, dim=-1)

    def _run_once(save_tokens: bool = False) -> float:
        weights["k_cache"].zero_()
        weights["v_cache"].zero_()
        pos_id.zero_()
        for pos in range(prompt_len):
            argmax = _decode_step(pos, prompt_tokens[pos])
        torch.cuda.synchronize()

        if save_tokens:
            output_tokens[0] = argmax[0]

        t0 = time.perf_counter()
        for i in range(num_decode_tokens):
            argmax = _decode_step(prompt_len + i, argmax)
            if save_tokens:
                output_tokens[i + 1] = argmax[0]
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    print("Compiling / first run...")
    _run_once()

    print(f"Warming up ({warmup} runs)...")
    for _ in range(warmup):
        _run_once()

    aggregate_tokens_per_sec_list = []
    per_seq_tokens_per_sec_list = []
    for sample in range(num_samples):
        decode_time = _run_once(save_tokens=(sample == num_samples - 1))
        aggregate_tok_sec = (batch_size * num_decode_tokens) / decode_time
        per_seq_tok_sec = num_decode_tokens / decode_time
        aggregate_tokens_per_sec_list.append(aggregate_tok_sec)
        per_seq_tokens_per_sec_list.append(per_seq_tok_sec)
        bandwidth_gbs = model_size * per_seq_tok_sec / 1e9
        flops_tfs = params * per_seq_tok_sec * 2 / 1e12
        print(
            f"Time for inference {sample + 1}: {decode_time:.02f} sec total, "
            f"{per_seq_tok_sec:.02f} tok/s/seq ({aggregate_tok_sec:.02f} aggregate tok/s)"
        )
        print(f"Bandwidth achieved: {bandwidth_gbs:.02f} GB/s")
        print(f"FLOPS achieved: {flops_tfs:.02f} TF/s")
        print()

    all_ids = torch.cat([prompt_ids, output_tokens])
    print(tokenizer.decode(all_ids.tolist()))
    print()

    print("==========")
    print(f"Batch size: {batch_size}")
    print(f"Layers: {num_layers}")
    print(f"Prompt tokens: {prompt_len}")
    print(f"Generated tokens per sequence: {max_new_tokens}")
    print(f"Pages per sequence: {pages_per_seq}")
    print(f"Pages per layer: {num_pages}")
    print(f"Average decode tok/s/seq: {torch.mean(torch.tensor(per_seq_tokens_per_sec_list)).item():.2f}")
    print(f"Average aggregate decode tok/s: {torch.mean(torch.tensor(aggregate_tokens_per_sec_list)).item():.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == "__main__":
    initialize_cuda_context()
    torch._dynamo.reset()

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Hello, my name is")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    args = parser.parse_args()

    NUM_LAYERS = args.num_layers

    compiled = megakittens.compile(decode, use_jit_cache=False, verbose=False, save_schedule=False)

    benchmark_tok_per_sec(
        compiled,
        num_layers=NUM_LAYERS,
        prompt=args.prompt,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        warmup=args.warmup,
    )
