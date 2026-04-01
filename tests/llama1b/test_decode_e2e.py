"""
End-to-end Llama-1B decode test.
Loads real HF weights, runs one megakernel decode step,
compares logits against both the PyTorch reference and HF.
"""

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from megakittens.jit.cuda_utils import get_sm_count, initialize_cuda_context
from megakittens.dispatcher import Dispatcher, ScalarField
from megakittens.llama1b.scheduler import (
    T, HIDDEN_DIM, INTERMEDIATE_DIM,
    NUM_ATTENTION_HEADS, NUM_KV_HEADS, HEAD_DIM, VOCAB_SIZE, RMS_NORM_EPS,
    MAX_SEQ_LEN, schedule_decode,
)

from .test_hf_reference import (
    MODEL_NAME, DEVICE, DTYPE,
    stack_weights, make_rope_table, decode_step, prefill_kv_cache,
)

initialize_cuda_context()

SCALARS = [
    ScalarField("pos_id", "unsigned int", 4, 4),
    ScalarField("attn_scale", "float", 4, 4),
    ScalarField("rms_norm_eps", "float", 4, 4),
]


def mk_decode_step(weights, hidden_states, k_cache, v_cache, rope_cos, rope_sin, pos_id):
    """Run one decode step through the megakernel."""
    sm_count = get_sm_count()
    schedule = schedule_decode(sm_count=sm_count)
    instruction_metas, tensor_metas, instructions, num_barriers, input_indices, output_indices = schedule

    dispatcher = Dispatcher(
        instruction_metas, tensor_metas, instructions, num_barriers,
        input_indices, output_indices,
        use_jit_cache=False,
        scalar_fields=SCALARS,
    )

    attn_scale = 1.0 / math.sqrt(HEAD_DIM)
    tensors = [
        weights["qkv_weights"],
        weights["o_weights"],
        weights["attn_norm_weights"],
        weights["mlp_norm_weights"],
        weights["up_weights"],
        weights["gate_weights"],
        weights["down_weights"],
        weights["lm_head_norm_weight"],
        weights["lm_head_weight"],
        hidden_states,
        torch.zeros(NUM_ATTENTION_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE),
        torch.zeros(HIDDEN_DIM, device=DEVICE, dtype=DTYPE),
        torch.zeros(INTERMEDIATE_DIM, device=DEVICE, dtype=DTYPE),
        torch.zeros(VOCAB_SIZE, device=DEVICE, dtype=DTYPE),
        k_cache,
        v_cache,
        rope_cos,
        rope_sin,
    ]

    logits = dispatcher(*tensors, pos_id, attn_scale, RMS_NORM_EPS)
    torch.cuda.synchronize()
    return logits


@torch.inference_mode()
def test_decode_e2e():
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = hf_model.config

    weights = stack_weights(hf_model)
    rope_cos, rope_sin = make_rope_table(config, MAX_SEQ_LEN, DEVICE)

    prompt = "The cat sat on"
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(DEVICE)
    prompt_len = input_ids.shape[1]
    print(f"Prompt: {prompt!r}, {prompt_len} tokens")

    # HF prefill + decode
    hf_prefill = hf_model(input_ids)
    next_token = hf_prefill.logits[0, -1].argmax().item()
    hf_decode = hf_model(torch.tensor([[next_token]], device=DEVICE), past_key_values=hf_prefill.past_key_values)
    hf_logits = hf_decode.logits[0, -1]
    print(f"HF: {tokenizer.decode([next_token])!r} -> {tokenizer.decode([hf_logits.argmax().item()])!r}")

    # Prefill our KV cache
    k_cache = torch.zeros(16, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v_cache = torch.zeros(16, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    prefill_kv_cache(input_ids[0].tolist(), weights, k_cache, v_cache, rope_cos, rope_sin)

    # PyTorch reference decode
    ref_k, ref_v = k_cache.clone(), v_cache.clone()
    ref_logits = decode_step(next_token, prompt_len, weights, ref_k, ref_v, rope_cos, rope_sin)

    # Megakernel decode
    mk_k, mk_v = k_cache.clone(), v_cache.clone()
    mk_hidden = weights["embed_weight"][next_token].clone()
    mk_logits = mk_decode_step(weights, mk_hidden, mk_k, mk_v, rope_cos, rope_sin, prompt_len)

    # MK vs PyTorch reference
    ref_diff = (mk_logits.float() - ref_logits.float()).abs()
    print(f"MK vs ref:  max_diff={ref_diff.max().item():.4f}, mean_diff={ref_diff.mean().item():.6f}")
    assert ref_diff.mean().item() < 0.1, f"MK vs ref mean diff too large: {ref_diff.mean().item()}"

    # MK vs HF
    hf_diff = (mk_logits.float() - hf_logits.float()).abs()
    print(f"MK vs HF:   max_diff={hf_diff.max().item():.4f}, mean_diff={hf_diff.mean().item():.6f}")
    assert hf_diff.mean().item() < 0.1, f"MK vs HF mean diff too large: {hf_diff.mean().item()}"

    # Top token agreement
    mk_top = mk_logits.argmax().item()
    ref_top = ref_logits.argmax().item()
    hf_top = hf_logits.argmax().item()
    print(f"Top token — MK: {tokenizer.decode([mk_top])!r}, ref: {tokenizer.decode([ref_top])!r}, HF: {tokenizer.decode([hf_top])!r}")
    assert mk_top == ref_top, f"MK vs ref token mismatch: {mk_top} vs {ref_top}"
    assert mk_top == hf_top, f"MK vs HF token mismatch: {mk_top} vs {hf_top}"

    print("PASS")


if __name__ == "__main__":
    test_decode_e2e()
