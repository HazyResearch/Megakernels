"""
Reference Llama-3.2-1B in pure PyTorch.
Loads real weights, runs prefill + decode, checks shapes and correctness against HF.
"""

import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda"
DTYPE = torch.bfloat16

# llama 3.2 1B config
NUM_LAYERS = 16
HIDDEN_DIM = 2048
INTERMEDIATE_DIM = 8192
NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64
VOCAB_SIZE = 128256
RMS_NORM_EPS = 1e-5


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    normed = x_float * torch.rsqrt(variance + eps)
    return (weight * normed).to(x.dtype)


def apply_rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return (x * cos + rotated * sin).to(x.dtype)


def _interleave_indices(num_heads: int, head_dim: int) -> torch.Tensor:
    # permutes [d0..d31, d32..d63] -> [d0, d32, d1, d33, ...] per head
    # mirrors interleave_rope() in megakernelsllama
    half = head_dim // 2
    indices = []
    for h in range(num_heads):
        offset = h * head_dim
        for i in range(half):
            indices.append(offset + i)
            indices.append(offset + half + i)
    return torch.tensor(indices)


def stack_weights(hf_model):
    # Q/K rows permuted to interleaved RoPE order, V unchanged
    config = hf_model.config
    model = hf_model.model
    layers = model.layers

    q_indices = _interleave_indices(config.num_attention_heads, config.head_dim)
    k_indices = _interleave_indices(config.num_key_value_heads, config.head_dim)

    qkv_weights = []
    o_weights = []
    attn_norm_weights = []
    mlp_norm_weights = []
    up_weights = []
    gate_weights = []
    down_weights = []

    for layer in layers:
        attn = layer.self_attn
        mlp = layer.mlp

        q_w = attn.q_proj.weight[q_indices]  # [2048, 2048]
        k_w = attn.k_proj.weight[k_indices]  # [512, 2048]
        v_w = attn.v_proj.weight              # [512, 2048]
        qkv = torch.cat([q_w, k_w, v_w], dim=0)  # [3072, 2048]
        qkv_weights.append(qkv)

        o_weights.append(attn.o_proj.weight)                    # [2048, 2048]
        attn_norm_weights.append(layer.input_layernorm.weight)  # [2048]
        mlp_norm_weights.append(layer.post_attention_layernorm.weight)  # [2048]
        up_weights.append(mlp.up_proj.weight)    # [8192, 2048]
        gate_weights.append(mlp.gate_proj.weight)  # [8192, 2048]
        down_weights.append(mlp.down_proj.weight)  # [2048, 8192]

    return {
        "qkv_weights": torch.stack(qkv_weights),            # [16, 3072, 2048]
        "o_weights": torch.stack(o_weights),                 # [16, 2048, 2048]
        "attn_norm_weights": torch.stack(attn_norm_weights), # [16, 2048]
        "mlp_norm_weights": torch.stack(mlp_norm_weights),   # [16, 2048]
        "up_weights": torch.stack(up_weights),               # [16, 8192, 2048]
        "gate_weights": torch.stack(gate_weights),           # [16, 8192, 2048]
        "down_weights": torch.stack(down_weights),           # [16, 2048, 8192]
        "lm_head_norm_weight": model.norm.weight,            # [2048]
        "lm_head_weight": hf_model.lm_head.weight,          # [128256, 2048]
        "embed_weight": model.embed_tokens.weight,           # [128256, 2048]
    }


def make_rope_table(config: LlamaConfig, max_seq_len: int, device):
    # generate cos/sin via HF, then reindex to interleaved layout
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    rope = LlamaRotaryEmbedding(config=config)
    positions = torch.arange(max_seq_len).unsqueeze(0)
    dummy = torch.empty(0, config.hidden_size, dtype=torch.float32)
    cos_hf, sin_hf = rope(dummy, positions)
    cos_hf = cos_hf.squeeze(0).to(device)  # [max_seq_len, head_dim]
    sin_hf = sin_hf.squeeze(0).to(device)

    one_head_indices = _interleave_indices(1, config.head_dim)
    return cos_hf[..., one_head_indices], sin_hf[..., one_head_indices]


@torch.inference_mode()
def decode_step(
    token_id: int,
    pos_id: int,
    weights: dict,
    k_cache: torch.Tensor,  # [num_layers, max_seq_len, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [num_layers, max_seq_len, num_kv_heads, head_dim]
    rope_cos: torch.Tensor,  # [max_seq_len, head_dim]
    rope_sin: torch.Tensor,  # [max_seq_len, head_dim]
) -> torch.Tensor:
    seq_len = pos_id + 1
    attn_scale = 1.0 / math.sqrt(HEAD_DIM)
    cos = rope_cos[pos_id]  # [64]
    sin = rope_sin[pos_id]  # [64]

    # embedding
    x = weights["embed_weight"][token_id]  # [2048]
    assert x.shape == (HIDDEN_DIM,)

    for layer_idx in range(NUM_LAYERS):
        # opcode 1: rmsnorm + qkv matvec + rope + kv cache append
        normed = rmsnorm(x, weights["attn_norm_weights"][layer_idx], RMS_NORM_EPS)
        assert normed.shape == (HIDDEN_DIM,)

        qkv = weights["qkv_weights"][layer_idx] @ normed  # [3072, 2048] @ [2048] -> [3072]
        q = qkv[:NUM_ATTENTION_HEADS * HEAD_DIM].view(NUM_ATTENTION_HEADS, HEAD_DIM)  # [32, 64]
        k = qkv[NUM_ATTENTION_HEADS * HEAD_DIM:(NUM_ATTENTION_HEADS + NUM_KV_HEADS) * HEAD_DIM].view(NUM_KV_HEADS, HEAD_DIM)  # [8, 64]
        v = qkv[(NUM_ATTENTION_HEADS + NUM_KV_HEADS) * HEAD_DIM:].view(NUM_KV_HEADS, HEAD_DIM)  # [8, 64]

        q = apply_rope_interleaved(q, cos, sin)
        k = apply_rope_interleaved(k, cos, sin)

        k_cache[layer_idx, pos_id] = k  # write [8, 64] into cache
        v_cache[layer_idx, pos_id] = v

        # opcode 2: partial attention (single partition)
        attn_out = torch.zeros(NUM_ATTENTION_HEADS, HEAD_DIM, device=x.device, dtype=x.dtype)  # [32, 64]
        for kv_head in range(NUM_KV_HEADS):
            k_cached = k_cache[layer_idx, :seq_len, kv_head]  # [seq_len, 64]
            v_cached = v_cache[layer_idx, :seq_len, kv_head]  # [seq_len, 64]
            gqa_start = kv_head * (NUM_ATTENTION_HEADS // NUM_KV_HEADS)
            gqa_end = gqa_start + (NUM_ATTENTION_HEADS // NUM_KV_HEADS)
            for q_head in range(gqa_start, gqa_end):
                scores = (q[q_head] @ k_cached.T) * attn_scale  # [64] @ [64, seq_len] -> [seq_len]
                w = F.softmax(scores.float(), dim=-1).to(x.dtype)  # [seq_len]
                attn_out[q_head] = w @ v_cached  # [seq_len] @ [seq_len, 64] -> [64]

        attn_out_flat = attn_out.reshape(HIDDEN_DIM)  # [2048]

        # opcode 3: attention reduction (skipped, single partition)

        # opcode 4: o projection + residual
        o_proj = weights["o_weights"][layer_idx] @ attn_out_flat  # [2048, 2048] @ [2048] -> [2048]
        x = x + o_proj

        # opcode 5: rmsnorm + up/gate matvec + silu gating
        normed_mlp = rmsnorm(x, weights["mlp_norm_weights"][layer_idx], RMS_NORM_EPS)
        gate = weights["gate_weights"][layer_idx] @ normed_mlp  # [8192, 2048] @ [2048] -> [8192]
        up = weights["up_weights"][layer_idx] @ normed_mlp      # [8192, 2048] @ [2048] -> [8192]
        silu_out = F.silu(gate) * up  # [8192]

        # opcode 6: down projection + residual
        down = weights["down_weights"][layer_idx] @ silu_out  # [2048, 8192] @ [8192] -> [2048]
        x = x + down

    # opcode 7: rmsnorm + lm head
    normed_final = rmsnorm(x, weights["lm_head_norm_weight"], RMS_NORM_EPS)
    logits = weights["lm_head_weight"] @ normed_final  # [128256, 2048] @ [2048] -> [128256]
    assert logits.shape == (VOCAB_SIZE,)

    return logits


def prefill_kv_cache(token_ids, weights, k_cache, v_cache, rope_cos, rope_sin):
    """Prefill KV cache one token at a time using the PyTorch reference."""
    attn_scale = 1.0 / math.sqrt(HEAD_DIM)
    for pos in range(len(token_ids)):
        x = weights["embed_weight"][token_ids[pos]]
        cos = rope_cos[pos]
        sin = rope_sin[pos]
        seq_len = pos + 1

        for layer_idx in range(NUM_LAYERS):
            normed = rmsnorm(x, weights["attn_norm_weights"][layer_idx], RMS_NORM_EPS)
            qkv = weights["qkv_weights"][layer_idx] @ normed
            q = qkv[:NUM_ATTENTION_HEADS * HEAD_DIM].view(NUM_ATTENTION_HEADS, HEAD_DIM)
            k = qkv[NUM_ATTENTION_HEADS * HEAD_DIM:(NUM_ATTENTION_HEADS + NUM_KV_HEADS) * HEAD_DIM].view(NUM_KV_HEADS, HEAD_DIM)
            v = qkv[(NUM_ATTENTION_HEADS + NUM_KV_HEADS) * HEAD_DIM:].view(NUM_KV_HEADS, HEAD_DIM)

            q = apply_rope_interleaved(q, cos, sin)
            k = apply_rope_interleaved(k, cos, sin)
            k_cache[layer_idx, pos] = k
            v_cache[layer_idx, pos] = v

            attn_out = torch.zeros(NUM_ATTENTION_HEADS, HEAD_DIM, device=x.device, dtype=x.dtype)
            for kv_head in range(NUM_KV_HEADS):
                k_cached = k_cache[layer_idx, :seq_len, kv_head]
                v_cached = v_cache[layer_idx, :seq_len, kv_head]
                gqa_size = NUM_ATTENTION_HEADS // NUM_KV_HEADS
                for q_head in range(kv_head * gqa_size, (kv_head + 1) * gqa_size):
                    scores = (q[q_head] @ k_cached.T) * attn_scale
                    if seq_len > 1:
                        mask = torch.full((seq_len,), float("-inf"), device=x.device)
                        mask[:pos + 1] = 0.0
                        scores = scores + mask
                    w = F.softmax(scores.float(), dim=-1).to(x.dtype)
                    attn_out[q_head] = w @ v_cached

            o_proj = weights["o_weights"][layer_idx] @ attn_out.reshape(HIDDEN_DIM)
            x = x + o_proj

            normed_mlp = rmsnorm(x, weights["mlp_norm_weights"][layer_idx], RMS_NORM_EPS)
            gate = weights["gate_weights"][layer_idx] @ normed_mlp
            up = weights["up_weights"][layer_idx] @ normed_mlp
            down = weights["down_weights"][layer_idx] @ (F.silu(gate) * up)
            x = x + down


@torch.inference_mode()
def test_llama1b_shapes_and_correctness():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    config = hf_model.config

    weights = stack_weights(hf_model)
    rope_cos, rope_sin = make_rope_table(config, 512, DEVICE)

    prompt = "The cat sat on"
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(DEVICE)
    prompt_len = input_ids.shape[1]
    print(f"Prompt: {prompt!r}, {prompt_len} tokens")

    # HF prefill
    hf_output = hf_model(input_ids)
    hf_next_token = hf_output.logits[0, -1].argmax().item()
    print(f"HF prefill -> {hf_next_token} = {tokenizer.decode([hf_next_token])!r}")

    # Our prefill
    k_cache = torch.zeros(NUM_LAYERS, 512, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v_cache = torch.zeros(NUM_LAYERS, 512, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    prefill_kv_cache(input_ids[0].tolist(), weights, k_cache, v_cache, rope_cos, rope_sin)

    # Decode: both us and HF predict the next token
    our_logits = decode_step(hf_next_token, prompt_len, weights, k_cache, v_cache, rope_cos, rope_sin)
    our_next_token = our_logits.argmax().item()
    print(f"Our decode -> {our_next_token} = {tokenizer.decode([our_next_token])!r}")

    hf_decode = hf_model(torch.tensor([[hf_next_token]], device=DEVICE), past_key_values=hf_output.past_key_values)
    hf_logits = hf_decode.logits[0, -1]
    hf_next = hf_logits.argmax().item()
    print(f"HF decode  -> {hf_next} = {tokenizer.decode([hf_next])!r}")

    assert our_next_token == hf_next, f"token mismatch: ours={our_next_token} vs HF={hf_next}"

    diff = (our_logits.float() - hf_logits.float()).abs()
    print(f"Logit diff: max={diff.max().item():.4f}, mean={diff.mean().item():.4f}")
    assert diff.max().item() < 2.0, f"logit max diff too large: {diff.max().item()}"
    print("PASS")


if __name__ == "__main__":
    test_llama1b_shapes_and_correctness()
