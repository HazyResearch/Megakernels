"""
End-to-end Llama-1B decode test.
Loads real HF weights, prefills KV cache with PyTorch reference,
runs one megakernel decode step, compares logits against PyTorch reference.
"""

import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from megakittens.jit.cuda_utils import get_sm_count, initialize_cuda_context
from megakittens.dispatcher import Dispatcher, ScalarField
from megakittens.llama1b.scheduler import (
    T, NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM,
    NUM_ATTENTION_HEADS, NUM_KV_HEADS, HEAD_DIM, VOCAB_SIZE, RMS_NORM_EPS,
    MAX_SEQ_LEN, schedule_decode,
)

initialize_cuda_context()

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda"
DTYPE = torch.bfloat16

SCALARS = [
    ScalarField("pos_id", "unsigned int", 4, 4),
    ScalarField("attn_scale", "float", 4, 4),
    ScalarField("rms_norm_eps", "float", 4, 4),
]


def rmsnorm(x, weight, eps):
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    return (x_f * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


def apply_rope_interleaved(x, cos, sin):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return (x * cos + rotated * sin).to(x.dtype)


def _interleave_indices(num_heads, head_dim):
    half = head_dim // 2
    indices = []
    for h in range(num_heads):
        offset = h * head_dim
        for i in range(half):
            indices.append(offset + i)
            indices.append(offset + half + i)
    return torch.tensor(indices)


def stack_weights(hf_model):
    config = hf_model.config
    model = hf_model.model
    layers = model.layers

    q_indices = _interleave_indices(config.num_attention_heads, config.head_dim)
    k_indices = _interleave_indices(config.num_key_value_heads, config.head_dim)

    qkv_weights, o_weights = [], []
    attn_norm_weights, mlp_norm_weights = [], []
    up_weights, gate_weights, down_weights = [], [], []

    for layer in layers:
        attn = layer.self_attn
        mlp = layer.mlp
        q_w = attn.q_proj.weight[q_indices]
        k_w = attn.k_proj.weight[k_indices]
        v_w = attn.v_proj.weight
        qkv_weights.append(torch.cat([q_w, k_w, v_w], dim=0))
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


def make_rope_table(config, max_seq_len, device):
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    rope = LlamaRotaryEmbedding(config=config)
    positions = torch.arange(max_seq_len).unsqueeze(0)
    dummy = torch.empty(0, config.hidden_size, dtype=torch.float32)
    cos_hf, sin_hf = rope(dummy, positions)
    cos_hf = cos_hf.squeeze(0).to(device)
    sin_hf = sin_hf.squeeze(0).to(device)
    one_head_indices = _interleave_indices(1, config.head_dim)
    return cos_hf[..., one_head_indices], sin_hf[..., one_head_indices]


def prefill_kv_cache(token_ids, weights, k_cache, v_cache, rope_cos, rope_sin):
    """Prefill KV cache one token at a time using PyTorch reference."""
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
                        mask = torch.full((seq_len,), float("-inf"), device=DEVICE)
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


def reference_decode_step(token_id, pos_id, weights, k_cache, v_cache, rope_cos, rope_sin):
    """PyTorch reference decode step."""
    seq_len = pos_id + 1
    attn_scale = 1.0 / math.sqrt(HEAD_DIM)
    cos = rope_cos[pos_id]
    sin = rope_sin[pos_id]

    x = weights["embed_weight"][token_id]

    for layer_idx in range(NUM_LAYERS):
        normed = rmsnorm(x, weights["attn_norm_weights"][layer_idx], RMS_NORM_EPS)
        qkv = weights["qkv_weights"][layer_idx] @ normed
        q = qkv[:NUM_ATTENTION_HEADS * HEAD_DIM].view(NUM_ATTENTION_HEADS, HEAD_DIM)
        k = qkv[NUM_ATTENTION_HEADS * HEAD_DIM:(NUM_ATTENTION_HEADS + NUM_KV_HEADS) * HEAD_DIM].view(NUM_KV_HEADS, HEAD_DIM)
        v = qkv[(NUM_ATTENTION_HEADS + NUM_KV_HEADS) * HEAD_DIM:].view(NUM_KV_HEADS, HEAD_DIM)

        q = apply_rope_interleaved(q, cos, sin)
        k = apply_rope_interleaved(k, cos, sin)
        k_cache[layer_idx, pos_id] = k
        v_cache[layer_idx, pos_id] = v

        attn_out = torch.zeros(NUM_ATTENTION_HEADS, HEAD_DIM, device=x.device, dtype=x.dtype)
        for kv_head in range(NUM_KV_HEADS):
            k_cached = k_cache[layer_idx, :seq_len, kv_head]
            v_cached = v_cache[layer_idx, :seq_len, kv_head]
            gqa_size = NUM_ATTENTION_HEADS // NUM_KV_HEADS
            for q_head in range(kv_head * gqa_size, (kv_head + 1) * gqa_size):
                scores = (q[q_head] @ k_cached.T) * attn_scale
                w = F.softmax(scores.float(), dim=-1).to(x.dtype)
                attn_out[q_head] = w @ v_cached

        o_proj = weights["o_weights"][layer_idx] @ attn_out.reshape(HIDDEN_DIM)
        x = x + o_proj

        normed_mlp = rmsnorm(x, weights["mlp_norm_weights"][layer_idx], RMS_NORM_EPS)
        gate = weights["gate_weights"][layer_idx] @ normed_mlp
        up = weights["up_weights"][layer_idx] @ normed_mlp
        down = weights["down_weights"][layer_idx] @ (F.silu(gate) * up)
        x = x + down

    normed_final = rmsnorm(x, weights["lm_head_norm_weight"], RMS_NORM_EPS)
    return weights["lm_head_weight"] @ normed_final


@torch.inference_mode()
def test_decode_e2e():
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = hf_model.config

    weights = stack_weights(hf_model)
    rope_cos, rope_sin = make_rope_table(config, MAX_SEQ_LEN, DEVICE)

    # Prefill
    prompt = "Tell me a joke about cookies."
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0].to(DEVICE)
    prompt_len = len(input_ids)
    print(f"Prompt: {prompt!r}, {prompt_len} tokens")

    k_cache = torch.zeros(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v_cache = torch.zeros(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    prefill_kv_cache(input_ids.tolist(), weights, k_cache, v_cache, rope_cos, rope_sin)

    # HF prefill to get next token
    hf_output = hf_model(input_ids.unsqueeze(0))
    next_token = hf_output.logits[0, -1].argmax().item()
    print(f"Next token: {next_token} = {tokenizer.decode([next_token])!r}")

    # PyTorch reference decode
    ref_k_cache = k_cache.clone()
    ref_v_cache = v_cache.clone()
    ref_logits = reference_decode_step(
        next_token, prompt_len, weights, ref_k_cache, ref_v_cache, rope_cos, rope_sin,
    )

    # Megakernel decode
    sm_count = get_sm_count()
    schedule = schedule_decode(sm_count=sm_count)
    instruction_metas, tensor_metas, instructions, num_barriers, input_indices, output_indices = schedule

    dispatcher = Dispatcher(
        instruction_metas, tensor_metas, instructions, num_barriers,
        input_indices, output_indices,
        use_jit_cache=False,
        scalar_fields=SCALARS,
    )

    mk_k_cache = k_cache.clone()
    mk_v_cache = v_cache.clone()
    hidden_states = weights["embed_weight"][next_token].clone()
    q_post_rope = torch.zeros(NUM_ATTENTION_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE)
    attn_out = torch.zeros(HIDDEN_DIM, device=DEVICE, dtype=DTYPE)
    silu_out = torch.zeros(INTERMEDIATE_DIM, device=DEVICE, dtype=DTYPE)
    logits = torch.zeros(VOCAB_SIZE, device=DEVICE, dtype=DTYPE)

    attn_scale = 1.0 / math.sqrt(HEAD_DIM)

    # Tensors in scheduler order (T.QKV_WEIGHTS=0, ..., T.ROPE_SIN=17)
    tensors = [
        weights["qkv_weights"],        # 0
        weights["o_weights"],           # 1
        weights["attn_norm_weights"],   # 2
        weights["mlp_norm_weights"],    # 3
        weights["up_weights"],          # 4
        weights["gate_weights"],        # 5
        weights["down_weights"],        # 6
        weights["lm_head_norm_weight"], # 7
        weights["lm_head_weight"],      # 8
        hidden_states,                  # 9
        q_post_rope,                    # 10
        attn_out,                       # 11
        silu_out,                       # 12
        logits,                         # 13
        mk_k_cache,                     # 14
        mk_v_cache,                     # 15
        rope_cos,                       # 16
        rope_sin,                       # 17
    ]

    mk_logits = dispatcher(*tensors, prompt_len, attn_scale, RMS_NORM_EPS)
    torch.cuda.synchronize()

    # Compare
    diff = (mk_logits.float() - ref_logits.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    mk_token = mk_logits.argmax().item()
    ref_token = ref_logits.argmax().item()

    mk_topk = mk_logits.topk(10).indices.tolist()
    ref_topk = ref_logits.topk(10).indices.tolist()

    print(f"max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f}")
    print(f"MK  top token: {mk_token} = {tokenizer.decode([mk_token])!r}")
    print(f"Ref top token: {ref_token} = {tokenizer.decode([ref_token])!r}")
    print(f"MK  top-10: {mk_topk}")
    print(f"Ref top-10: {ref_topk}")

    # For now, just print diagnostics — don't assert until we debug
    print("DONE")


if __name__ == "__main__":
    test_decode_e2e()
