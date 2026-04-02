"""
Head-to-head benchmark: old Megakernels-main vs new MegaKittens decode kernel.
Measures microseconds per decode step with random weights (no HF model needed).
"""

import math
import sys
import os
import time

import torch

# ── Constants ─────────────────────────────────────────────────
HIDDEN_DIM = 2048
NUM_LAYERS = 16
NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64
GQA_RATIO = NUM_ATTENTION_HEADS // NUM_KV_HEADS
QKV_DIM = (NUM_ATTENTION_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
Q_DIM = NUM_ATTENTION_HEADS * HEAD_DIM
K_DIM = NUM_KV_HEADS * HEAD_DIM
INTERMEDIATE_DIM = 8192
VOCAB_SIZE = 128256
MAX_SEQ_LEN = 512
BLOCK_SIZE = 16
SEQ_LEN = 256
DEVICE = "cuda"
DTYPE = torch.bfloat16

B300_BW = 8_000_000_000_000  # 8 TB/s


def _time_us(fn, warmup=100, iters=500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # microseconds


# ── New MegaKittens kernel ────────────────────────────────────

def bench_new():
    from megakittens.jit.cuda_utils import get_sm_count, initialize_cuda_context
    from megakittens.dispatcher import Dispatcher, ScalarField
    from megakittens.llama1b.scheduler import schedule_decode

    initialize_cuda_context()
    sm_count = get_sm_count()
    pos_id = SEQ_LEN - 1
    attn_scale = 1.0 / math.sqrt(HEAD_DIM)

    schedule = schedule_decode(sm_count=sm_count)
    instruction_metas, tensor_metas, instructions, num_barriers, input_indices, output_indices = schedule

    scalars = [
        ScalarField("pos_id", "unsigned int", 4, 4),
        ScalarField("attn_scale", "float", 4, 4),
        ScalarField("rms_norm_eps", "float", 4, 4),
    ]

    dispatcher = Dispatcher(
        instruction_metas, tensor_metas, instructions, num_barriers,
        input_indices, output_indices,
        use_jit_cache=False,
        scalar_fields=scalars,
    )

    D = DEVICE
    tensors = [
        torch.randn(NUM_LAYERS, QKV_DIM, HIDDEN_DIM, dtype=DTYPE, device=D),
        torch.randn(NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM, dtype=DTYPE, device=D),
        torch.randn(NUM_LAYERS, HIDDEN_DIM, dtype=DTYPE, device=D),
        torch.randn(NUM_LAYERS, HIDDEN_DIM, dtype=DTYPE, device=D),
        torch.randn(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=DTYPE, device=D),
        torch.randn(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=DTYPE, device=D),
        torch.randn(NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM, dtype=DTYPE, device=D),
        torch.randn(HIDDEN_DIM, dtype=DTYPE, device=D),
        torch.randn(VOCAB_SIZE, HIDDEN_DIM, dtype=DTYPE, device=D),
        torch.randn(HIDDEN_DIM, dtype=DTYPE, device=D),
        torch.zeros(Q_DIM, dtype=DTYPE, device=D),
        torch.zeros(HIDDEN_DIM, dtype=DTYPE, device=D),
        torch.zeros(INTERMEDIATE_DIM, dtype=DTYPE, device=D),
        torch.zeros(VOCAB_SIZE, dtype=DTYPE, device=D),
        torch.randn(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=D),
        torch.randn(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=D),
        torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=D),
        torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=D),
    ]

    fn = lambda: dispatcher(*tensors, pos_id, attn_scale, 1e-5)
    us = _time_us(fn)
    return us


# ── Old Megakernels-main kernel ───────────────────────────────

def bench_old():
    old_root = "/home/stuart/enbao/Megakernels-main"
    sys.path.insert(0, old_root)
    os.chdir(old_root)

    from megakernels.dispatch import make_mk_interpreter, make_schedule_builder
    from megakernels.llama import LlamaForCausalLM
    from megakernels.model_types import ExtraModelConfig
    from megakernels.scheduler import assign_to_sms, tensorize_instructions
    from megakernels.demos.latency.mk import interpret_with_mk

    extra_config = ExtraModelConfig(
        interleave_rope=True,
        max_len_override=MAX_SEQ_LEN,
        max_batch_size=1,
    )
    # Build a dummy model to get the schedule infrastructure
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        device=DEVICE,
        extra_config=extra_config,
    )

    schedule_builder = make_schedule_builder("latency")
    schedule = schedule_builder.build(model)
    assigned_to_sms = assign_to_sms("rr", schedule=schedule)
    tensorize_instructions(schedule.globs, assigned_to_sms)

    from pathlib import Path
    mk_dir = Path(old_root) / "demos" / "low-latency-llama"
    interpreter = make_mk_interpreter("latency", mk_dir)

    globs = schedule.globs
    globs.pos_id = SEQ_LEN - 1
    globs.attn_scale = 1.0 / math.sqrt(HEAD_DIM)
    globs.rms_norm_eps = 1e-5
    globs.skip_attn_reduction = False

    # Debug: check schedule setup
    print(f"  Instructions shape: {globs.instructions.shape}")
    print(f"  Barriers shape: {globs.barriers.shape}")
    print(f"  Barriers nonzero: {globs.barriers.nonzero().shape[0]}")
    insts = globs.instructions
    print(f"  First few instruction opcodes: {insts[:5, 0].tolist()}")
    print(f"  Nonzero instruction rows: {(insts[:, 0] != 0).sum().item()}")

    # Populate hidden_states with non-zero data
    globs.hidden_states.normal_()
    print(f"  hidden_states sum before: {globs.hidden_states.float().sum().item():.4f}")

    # Verify it actually runs — single call + sync
    globs.barriers.zero_()
    interpret_with_mk(globs, interpreter.mk_func)
    torch.cuda.synchronize()
    err = torch.cuda.last_error()
    print(f"  CUDA error: {err}")
    print(f"  logits sum after: {globs.logits.float().sum().item():.2f}")
    print(f"  hidden_states sum after: {globs.hidden_states.float().sum().item():.4f}")

    def fn():
        globs.barriers.zero_()
        interpret_with_mk(globs, interpreter.mk_func)
        torch.cuda.synchronize()

    us = _time_us(fn)

    os.chdir("/home/stuart/enbao/LlamaKernels")
    return us


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    per_layer_bytes = (
        QKV_DIM * HIDDEN_DIM * 2
        + HIDDEN_DIM * 2
        + 2 * HEAD_DIM * 4
        + 2 * SEQ_LEN * NUM_KV_HEADS * HEAD_DIM * 2
        + HIDDEN_DIM * HIDDEN_DIM * 2
        + HIDDEN_DIM * 2
        + 2 * INTERMEDIATE_DIM * HIDDEN_DIM * 2
        + HIDDEN_DIM * INTERMEDIATE_DIM * 2
    )
    lm_head_bytes = HIDDEN_DIM * 2 + VOCAB_SIZE * HIDDEN_DIM * 2 + VOCAB_SIZE * 2
    total_bytes = NUM_LAYERS * per_layer_bytes + lm_head_bytes
    roof_us = total_bytes / B300_BW * 1e6

    print("=" * 60)
    print("Llama 3.2 1B decode: old Megakernels vs new MegaKittens")
    print(f"seq_len={SEQ_LEN}, layers={NUM_LAYERS}, bf16")
    print(f"Model bytes: {total_bytes / 1e9:.2f} GB")
    print(f"B300 roofline: {roof_us:.1f} us")
    print("=" * 60)

    # Run old first (loads HF model)
    print("\nBenchmarking old Megakernels-main...")
    old_us = bench_old()

    print("\nBenchmarking new MegaKittens...")
    new_us = bench_new()

    old_gbs = total_bytes / (old_us * 1e-6) / 1e9
    new_gbs = total_bytes / (new_us * 1e-6) / 1e9

    print()
    print(f"  Old (Megakernels-main):  {old_us:>8.1f} us  ({old_gbs:.1f} GB/s)  {old_us/roof_us:.1f}x roof")
    print(f"  New (MegaKittens):       {new_us:>8.1f} us  ({new_gbs:.1f} GB/s)  {new_us/roof_us:.1f}x roof")
    print(f"  Speedup: {old_us/new_us:.2f}x")
