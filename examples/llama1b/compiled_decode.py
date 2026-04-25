"""Multi-layer decode using megakittens.compile on explicit fused llama1b ops.

The decode forward is written in plain PyTorch using the five fused custom ops,
slicing per-layer weights via `[i:i+1]` so the tracer narrows each op's
TensorRange. `megakittens.compile` runs the tracer + scheduler + dispatcher
end-to-end instead of the hand-written schedule in `scheduler.py`.
"""

from __future__ import annotations

import ctypes
import math
import time

import cuda.bindings.driver as cuda_driver
import torch

import megakittens
from megakittens.jit.cuda_utils import initialize_cuda_context
from .benchmark_instructions import (
    _make_rope_table,
    _prefill_kv_cache,
    _rmsnorm,
    _stack_weights,
)
from .scheduler import (
    HEAD_DIM,
    HIDDEN_DIM,
    INTERMEDIATE_DIM,
    MAX_SEQ_LEN,
    NUM_KV_HEADS,
    NUM_LAYERS,
    RMS_NORM_EPS,
    VOCAB_SIZE,
)

ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)
DECODE_NUM_LAYERS = NUM_LAYERS


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
    for i in range(DECODE_NUM_LAYERS):
        layer_idx = i % NUM_LAYERS
        q = torch.ops.megakittens.rms_qkv_rope_append(
            hidden_states,
            attn_norm_weights[layer_idx:layer_idx+1],
            qkv_weights[layer_idx:layer_idx+1],
            rope_cos,
            rope_sin,
            k_cache[layer_idx:layer_idx+1],
            v_cache[layer_idx:layer_idx+1],
            pos_id,
            rms_norm_eps,
        )

        attn_out = torch.ops.megakittens.attention_partial(
            q, k_cache[layer_idx:layer_idx+1], v_cache[layer_idx:layer_idx+1], pos_id, attn_scale,
        )

        # o_proj + residual: mutates hidden_states in place
        torch.ops.megakittens.mat_vec_adds(hidden_states, attn_out, o_weights[layer_idx:layer_idx+1])

        silu_out = torch.ops.megakittens.rms_upgate_silu(
            hidden_states,
            mlp_norm_weights[layer_idx:layer_idx+1],
            up_weights[layer_idx:layer_idx+1],
            gate_weights[layer_idx:layer_idx+1],
            rms_norm_eps,
        )

        torch.ops.megakittens.mat_vec_adds(
            hidden_states, silu_out, down_weights[layer_idx:layer_idx+1],
        )

    logits = torch.ops.megakittens.rms_lm_head(
        hidden_states, lm_head_norm_weight, lm_head_weight, rms_norm_eps,
    )
    return logits


def decode_compile_individual_ops():
    def _compile(fn):
        return megakittens.compile(
            fn,
            use_jit_cache=True,
            verbose=False,
            save_schedule=False,
            cluster_size=1,
            instruction_pipeline_stages=2,
            no_inter_op_inst_overlap=False,
            no_inst_overlap=False,
            coarse_grained_barriers=False,
        )

    def qkv_op(
        hidden_states,
        attn_norm_weights,
        qkv_weights,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        pos_id,
        rms_norm_eps,
    ):
        return torch.ops.megakittens.rms_qkv_rope_append(
            hidden_states,
            attn_norm_weights,
            qkv_weights,
            rope_cos,
            rope_sin,
            k_cache,
            v_cache,
            pos_id,
            rms_norm_eps,
        )

    def attention_op(q, k_cache, v_cache, pos_id, attn_scale):
        return torch.ops.megakittens.attention_partial(q, k_cache, v_cache, pos_id, attn_scale)

    def matvec_adds_op(residual, x, weights):
        torch.ops.megakittens.mat_vec_adds(residual, x, weights)
        return residual

    def upgate_op(hidden_states, mlp_norm_weights, up_weights, gate_weights, rms_norm_eps):
        return torch.ops.megakittens.rms_upgate_silu(
            hidden_states,
            mlp_norm_weights,
            up_weights,
            gate_weights,
            rms_norm_eps,
        )

    def lm_head_op(hidden_states, lm_head_norm_weight, lm_head_weight, rms_norm_eps):
        return torch.ops.megakittens.rms_lm_head(
            hidden_states,
            lm_head_norm_weight,
            lm_head_weight,
            rms_norm_eps,
        )

    compiled_qkv = _compile(qkv_op)
    compiled_attention = _compile(attention_op)
    compiled_o_proj = _compile(matvec_adds_op)
    compiled_upgate = _compile(upgate_op)
    compiled_down_proj = _compile(matvec_adds_op)
    compiled_lm_head = _compile(lm_head_op)

    def compiled(
        hidden_states,
        qkv_weights,
        o_weights,
        attn_norm_weights,
        mlp_norm_weights,
        up_weights,
        gate_weights,
        down_weights,
        lm_head_norm_weight,
        lm_head_weight,
        k_cache,
        v_cache,
        rope_cos,
        rope_sin,
        pos_id,
        attn_scale,
        rms_norm_eps,
    ):
        for i in range(DECODE_NUM_LAYERS):
            layer_idx = i % NUM_LAYERS
            q = compiled_qkv(
                hidden_states,
                attn_norm_weights[layer_idx:layer_idx+1],
                qkv_weights[layer_idx:layer_idx+1],
                rope_cos,
                rope_sin,
                k_cache[layer_idx:layer_idx+1],
                v_cache[layer_idx:layer_idx+1],
                pos_id,
                rms_norm_eps,
            )

            attn_out = compiled_attention(
                q, k_cache[layer_idx:layer_idx+1], v_cache[layer_idx:layer_idx+1], pos_id, attn_scale,
            )

            hidden_states = compiled_o_proj(
                hidden_states, attn_out, o_weights[layer_idx:layer_idx+1],
            )

            silu_out = compiled_upgate(
                hidden_states,
                mlp_norm_weights[layer_idx:layer_idx+1],
                up_weights[layer_idx:layer_idx+1],
                gate_weights[layer_idx:layer_idx+1],
                rms_norm_eps,
            )

            hidden_states = compiled_down_proj(
                hidden_states, silu_out, down_weights[layer_idx:layer_idx+1],
            )

        return compiled_lm_head(
            hidden_states, lm_head_norm_weight, lm_head_weight, rms_norm_eps,
        )

    return compiled


def decode_compile_per_layer():
    def _compile(fn):
        return megakittens.compile(
            fn,
            use_jit_cache=True,
            verbose=False,
            save_schedule=False,
            cluster_size=1,
            instruction_pipeline_stages=2,
            no_inter_op_inst_overlap=False,
            no_inst_overlap=False,
            coarse_grained_barriers=False,
        )

    def layer_op(
        hidden_states,
        qkv_weights,
        o_weights,
        attn_norm_weights,
        mlp_norm_weights,
        up_weights,
        gate_weights,
        down_weights,
        k_cache,
        v_cache,
        rope_cos,
        rope_sin,
        pos_id,
        attn_scale,
        rms_norm_eps,
    ):
        q = torch.ops.megakittens.rms_qkv_rope_append(
            hidden_states,
            attn_norm_weights,
            qkv_weights,
            rope_cos,
            rope_sin,
            k_cache,
            v_cache,
            pos_id,
            rms_norm_eps,
        )

        attn_out = torch.ops.megakittens.attention_partial(
            q, k_cache, v_cache, pos_id, attn_scale,
        )

        torch.ops.megakittens.mat_vec_adds(hidden_states, attn_out, o_weights)

        silu_out = torch.ops.megakittens.rms_upgate_silu(
            hidden_states,
            mlp_norm_weights,
            up_weights,
            gate_weights,
            rms_norm_eps,
        )

        torch.ops.megakittens.mat_vec_adds(hidden_states, silu_out, down_weights)
        return hidden_states

    def lm_head_op(hidden_states, lm_head_norm_weight, lm_head_weight, rms_norm_eps):
        return torch.ops.megakittens.rms_lm_head(
            hidden_states,
            lm_head_norm_weight,
            lm_head_weight,
            rms_norm_eps,
        )

    compiled_layer = _compile(layer_op)
    compiled_lm_head = _compile(lm_head_op)

    def compiled(
        hidden_states,
        qkv_weights,
        o_weights,
        attn_norm_weights,
        mlp_norm_weights,
        up_weights,
        gate_weights,
        down_weights,
        lm_head_norm_weight,
        lm_head_weight,
        k_cache,
        v_cache,
        rope_cos,
        rope_sin,
        pos_id,
        attn_scale,
        rms_norm_eps,
    ):
        for i in range(DECODE_NUM_LAYERS):
            layer_idx = i % NUM_LAYERS
            hidden_states = compiled_layer(
                hidden_states,
                qkv_weights[layer_idx:layer_idx+1],
                o_weights[layer_idx:layer_idx+1],
                attn_norm_weights[layer_idx:layer_idx+1],
                mlp_norm_weights[layer_idx:layer_idx+1],
                up_weights[layer_idx:layer_idx+1],
                gate_weights[layer_idx:layer_idx+1],
                down_weights[layer_idx:layer_idx+1],
                k_cache[layer_idx:layer_idx+1],
                v_cache[layer_idx:layer_idx+1],
                rope_cos,
                rope_sin,
                pos_id,
                attn_scale,
                rms_norm_eps,
            )

        return compiled_lm_head(
            hidden_states, lm_head_norm_weight, lm_head_weight, rms_norm_eps,
        )

    return compiled


@torch.inference_mode()
def benchmark_tok_per_sec(
    prompt="Hello, my name is",
    max_new_tokens=200,
    num_samples=5,
    warmup=5,
    compile_individual_ops=False,
    compile_per_layer=False,
    no_input_barriers=False,
    no_reuse_barriers=False,
    num_layers=NUM_LAYERS,
):
    """tok/s with HF weights + greedy decode, using megakittens.compile(decode)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    D = "cuda"
    if num_layers < 1:
        raise RuntimeError(f"num_layers must be >= 1, got {num_layers}")
    if compile_individual_ops and compile_per_layer:
        raise RuntimeError("compile_individual_ops and compile_per_layer are mutually exclusive")
    if (no_input_barriers or no_reuse_barriers) and (compile_individual_ops or compile_per_layer):
        raise RuntimeError("barrier ablations are only supported in whole-compile mode")
    global DECODE_NUM_LAYERS
    DECODE_NUM_LAYERS = num_layers

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

    print("Attention mode: no reduction")

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

    if compile_individual_ops:
        compiled = decode_compile_individual_ops()
    elif compile_per_layer:
        compiled = decode_compile_per_layer()
    else:
        compiled = megakittens.compile(
            decode,
            use_jit_cache=False,
            verbose=False,
            save_schedule=False,
            cluster_size=1,
            instruction_pipeline_stages=2,
            no_inter_op_inst_overlap=False,
            no_inst_overlap=False,
            no_input_barriers=no_input_barriers,
            no_reuse_barriers=no_reuse_barriers,
            coarse_grained_barriers=False
        )

    # Pre-allocate CPU-side buffer and cache GPU address for fast pos_id updates
    _pos_id_buf = (ctypes.c_int * 1)(0)
    _pos_id_gpu_ptr = pos_id_tensor.data_ptr()

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

    def _decode_step(pos_id, input_token):
        hidden_states.copy_(embedding(input_token))
        _pos_id_buf[0] = pos_id
        stream = torch.cuda.current_stream().cuda_stream
        cuda_driver.cuMemcpyHtoDAsync(_pos_id_gpu_ptr, _pos_id_buf, 4, stream)
        logits = compiled(*decode_args)
        return torch.argmax(logits, dim=-1)

    compiled(*decode_args)
    output_tokens[0] = first_token
    print(f"Warming up ({warmup} runs)...")
    for _ in range(warmup):
        k_cache.copy_(k_cache_snapshot)
        v_cache.copy_(v_cache_snapshot)
        token = first_token
        for i in range(num_decode_tokens):
            token = _decode_step(prompt_len + i, token)
            output_tokens[i + 1] = token
    torch.cuda.synchronize()
    print("Warmup done.")

    decode_tokens_per_sec_list = []
    for sample in range(num_samples):
        k_cache.copy_(k_cache_snapshot)
        v_cache.copy_(v_cache_snapshot)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        token = first_token
        for i in range(num_decode_tokens):
            token = _decode_step(prompt_len + i, token)
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
    print(f"Attention mode: no reduction")
    print(f"Average tokens/sec (decode only): {torch.mean(torch.tensor(decode_tokens_per_sec_list)).item():.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == "__main__":
    import argparse
    initialize_cuda_context()
    torch._dynamo.reset()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Tell me a joke about cookies.")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--compile-individual-ops", action="store_true")
    parser.add_argument("--compile-per-layer", action="store_true")
    parser.add_argument("--no-input-barriers", action="store_true")
    parser.add_argument("--no-reuse-barriers", action="store_true")
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    args = parser.parse_args()
    benchmark_tok_per_sec(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        warmup=args.warmup,
        compile_individual_ops=args.compile_individual_ops,
        compile_per_layer=args.compile_per_layer,
        no_input_barriers=args.no_input_barriers,
        no_reuse_barriers=args.no_reuse_barriers,
        num_layers=args.num_layers,
    )
