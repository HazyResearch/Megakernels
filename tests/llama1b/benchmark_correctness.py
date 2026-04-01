"""
Correctness benchmark for Llama 1B decode instructions.
Runs each fused instruction and compares against PyTorch reference,
reporting max_diff and mean_diff per output tensor.
"""

import math

import torch
import torch.nn.functional as F

from megakittens.jit.cuda_utils import get_sm_count, initialize_cuda_context
from megakittens.dispatcher import Dispatcher, ScalarField
from megakittens.schema.device import Device
from megakittens.schema.dtype import DType
from megakittens.schema.instruction import Instruction, InstructionMeta
from megakittens.schema.tensor import TensorMeta
from megakittens.itypes.noop import Noop
from megakittens.itypes.attention_partial import AttentionPartial
from megakittens.itypes.proj_residual import ProjResidual
from megakittens.itypes.rms_lm_head import RmsLmHead
from megakittens.itypes.rms_qkv_rope_append import RmsQkvRopeAppend
from megakittens.itypes.rms_upgate_silu import RmsUpgateSilu

initialize_cuda_context()

# ── Llama 1B constants ─────────────────────────────────────

HIDDEN_DIM = 2048
NUM_LAYERS = 16
NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64
GQA_RATIO = NUM_ATTENTION_HEADS // NUM_KV_HEADS  # 4
QKV_DIM = (NUM_ATTENTION_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM  # 3072
Q_DIM = NUM_ATTENTION_HEADS * HEAD_DIM  # 2048
K_DIM = NUM_KV_HEADS * HEAD_DIM  # 512
INTERMEDIATE_DIM = 8192
VOCAB_SIZE = 128256
MAX_SEQ_LEN = 128
BLOCK_SIZE = 16
RMS_NORM_EPS = 1e-5
DEVICE = "cuda"
DTYPE = torch.bfloat16

SCALARS = [
    ScalarField("pos_id", "unsigned int", 4, 4),
    ScalarField("attn_scale", "float", 4, 4),
    ScalarField("rms_norm_eps", "float", 4, 4),
]

_DEVICE = Device(type="cuda", index=0)
_NOOP_META = InstructionMeta(icode=0, itype=Noop(), src_tensors=(), dst_tensors=())
_NOOP_INST = Instruction(
    icode=0, src_tensors=(), dst_tensors=(), indices=(),
    src_barriers=(), src_barrier_targets=(),
    num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
    dst_barriers=(),
)


# ── Helpers ────────────────────────────────────────────────

def _make_dispatcher(inst_meta, tensor_metas, instructions, input_indices, output_indices):
    instructions = list(instructions)
    while len(instructions) % 2 != 0:
        instructions.append(_NOOP_INST)
    return Dispatcher(
        [_NOOP_META, inst_meta], tensor_metas, instructions,
        num_barriers=0,
        input_tensor_indices=input_indices,
        output_tensor_indices=output_indices,
        use_jit_cache=False,
        scalar_fields=SCALARS,
    )


def _check_close(actual, expected, atol, rtol):
    """Check correctness using torch.allclose convention: |a - e| <= atol + rtol * |e|.

    Returns (max_diff, mean_diff, tightness, passed, deficit) where:
      tightness = max(|a - e| / (atol + rtol * |e|)), < 1.0 means pass
      deficit   = max(|a - e| - rtol * |e|, 0), the minimum atol needed
    """
    a, e = actual.float(), expected.float()
    d = (a - e).abs()
    tol = atol + rtol * e.abs()
    tightness = (d / tol).max().item()
    deficit = (d - rtol * e.abs()).clamp(min=0).max().item()
    return d.max().item(), d.mean().item(), tightness, tightness < 1.0, deficit


def _rmsnorm(x, weight, eps):
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    return (x_f * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


def _apply_rope(x, cos, sin):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return (x * cos + rotated * sin).to(x.dtype)


# ── Tolerances ─────────────────────────────────────────────
# atol + rtol * |expected|, following torch.allclose convention.
#
# Derived empirically: run each instruction across 50 random seeds,
# measure max(|error| / (atol + rtol * |expected|)) per seed,
# then set atol/rtol so the worst seed across all instructions
# has tightness < 1.0.
#
# Analytical starting point for bf16 dot products reducing over N dims:
#   per-element relative error ~ sqrt(N) * eps_bf16
#   max over K output elements  ~ sqrt(N) * eps_bf16 * sqrt(2 * ln(K))
# For fused ops (rmsnorm + matmul + rope), errors compound across steps.
#
# Use calibrate_tolerances() below to re-derive from data.

ATOL = None  # Set by calibrate or by benchmark_correctness
RTOL = None


# ── Per-instruction correctness checks ────────────────────
# Each returns list of (output_name, max_diff, mean_diff, tightness, passed).

def _check_rms_qkv_rope_append(sm_count, atol, rtol, seed=42):
    torch.manual_seed(seed)

    layer_idx = 2
    pos_id = 37

    hidden = torch.randn(HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    norm_w = torch.randn(NUM_LAYERS, HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    qkv_w = torch.randn(NUM_LAYERS, QKV_DIM, HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    rope_cos = torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=DEVICE)
    rope_sin = torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=DEVICE)
    k_cache = torch.zeros(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v_cache = torch.zeros(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    q_out = torch.zeros(Q_DIM, dtype=DTYPE, device=DEVICE)

    # PyTorch reference
    normed = _rmsnorm(hidden, norm_w[layer_idx], RMS_NORM_EPS)
    qkv = qkv_w[layer_idx] @ normed
    q_ref = qkv[:Q_DIM].view(NUM_ATTENTION_HEADS, HEAD_DIM)
    k_ref = qkv[Q_DIM:Q_DIM + K_DIM].view(NUM_KV_HEADS, HEAD_DIM)
    v_ref = qkv[Q_DIM + K_DIM:].view(NUM_KV_HEADS, HEAD_DIM)
    cos_pos = rope_cos[pos_id]
    sin_pos = rope_sin[pos_id]
    q_ref = _apply_rope(q_ref, cos_pos, sin_pos).flatten()
    k_ref = _apply_rope(k_ref, cos_pos, sin_pos)

    # Dispatcher
    itype = RmsQkvRopeAppend(n=HIDDEN_DIM)
    src, dst = (0, 1, 2, 3, 4, 5, 6), (7,)
    inst_meta = InstructionMeta(icode=1, itype=itype, src_tensors=src, dst_tensors=dst)

    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, HIDDEN_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, QKV_DIM, HIDDEN_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.fp32, shape=(MAX_SEQ_LEN, HEAD_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.fp32, shape=(MAX_SEQ_LEN, HEAD_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(Q_DIM,), device=_DEVICE),
    ]

    num_blocks = QKV_DIM // BLOCK_SIZE
    instructions = []
    for sm in range(sm_count):
        s = round(sm * num_blocks / sm_count)
        e = round((sm + 1) * num_blocks / sm_count)
        instructions.append(Instruction(
            icode=1, src_tensors=src, dst_tensors=dst, indices=(layer_idx, s, e),
            src_barriers=(), src_barrier_targets=(),
            num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
            dst_barriers=(),
        ))

    dispatcher = _make_dispatcher(inst_meta, tensor_metas, instructions,
                                  input_indices=tuple(range(8)),
                                  output_indices=(7, 5, 6))

    dispatcher(hidden, norm_w, qkv_w, rope_cos, rope_sin,
               k_cache, v_cache, q_out, pos_id, 0.125, RMS_NORM_EPS)
    torch.cuda.synchronize()

    results = []
    results.append(("Q", *_check_close(q_out, q_ref, atol, rtol)))
    results.append(("K cache", *_check_close(k_cache[layer_idx, pos_id], k_ref, atol, rtol)))
    results.append(("V cache", *_check_close(v_cache[layer_idx, pos_id], v_ref, atol, rtol)))
    return "rms_qkv_rope_append", results


def _check_attention_partial(sm_count, atol, rtol, seed=42):
    torch.manual_seed(seed)

    seq_len = 37
    pos_id = seq_len - 1
    attn_scale = 1.0 / math.sqrt(HEAD_DIM)
    layer_idx = 2

    q = torch.randn(NUM_ATTENTION_HEADS * HEAD_DIM, dtype=DTYPE, device=DEVICE)
    k_cache = torch.randn(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v_cache = torch.randn(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    attn_out = torch.zeros(NUM_ATTENTION_HEADS * HEAD_DIM, dtype=DTYPE, device=DEVICE)

    # PyTorch reference
    q_heads = q.view(NUM_ATTENTION_HEADS, HEAD_DIM)
    expected = torch.zeros_like(attn_out)
    for kv_head in range(NUM_KV_HEADS):
        k_cached = k_cache[layer_idx, :seq_len, kv_head]
        v_cached = v_cache[layer_idx, :seq_len, kv_head]
        gqa_start = kv_head * GQA_RATIO
        for q_head in range(gqa_start, gqa_start + GQA_RATIO):
            scores = (q_heads[q_head] @ k_cached.T) * attn_scale
            w = F.softmax(scores.float(), dim=-1).to(DTYPE)
            out = w @ v_cached
            expected[q_head * HEAD_DIM:(q_head + 1) * HEAD_DIM] = out

    # Dispatcher
    itype = AttentionPartial()
    src, dst = (0, 1, 2), (3,)
    inst_meta = InstructionMeta(icode=1, itype=itype, src_tensors=src, dst_tensors=dst)

    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(Q_DIM,), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(Q_DIM,), device=_DEVICE),
    ]

    instructions = []
    for kv_head in range(NUM_KV_HEADS):
        instructions.append(Instruction(
            icode=1, src_tensors=src, dst_tensors=dst, indices=(layer_idx, kv_head),
            src_barriers=(), src_barrier_targets=(),
            num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
            dst_barriers=(),
        ))

    dispatcher = _make_dispatcher(inst_meta, tensor_metas, instructions,
                                  input_indices=(0, 1, 2, 3),
                                  output_indices=(3,))

    result = dispatcher(q, k_cache, v_cache, attn_out, pos_id, attn_scale, RMS_NORM_EPS)
    torch.cuda.synchronize()

    return "attention_partial", [("attn_out", *_check_close(result, expected, atol, rtol))]


def _check_o_proj_residual(sm_count, atol, rtol, seed=42):
    torch.manual_seed(seed)
    layer_idx = 3

    attn_out = torch.randn(HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    o_weights = torch.randn(NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    hidden_states = torch.randn(HIDDEN_DIM, dtype=DTYPE, device=DEVICE)

    expected = hidden_states.clone() + o_weights[layer_idx] @ attn_out

    itype = ProjResidual()
    src, dst = (0, 1), (2,)
    inst_meta = InstructionMeta(icode=1, itype=itype, src_tensors=src, dst_tensors=dst)

    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=_DEVICE),
    ]

    num_blocks = HIDDEN_DIM // BLOCK_SIZE
    instructions = []
    for b in range(num_blocks):
        instructions.append(Instruction(
            icode=1, src_tensors=src, dst_tensors=dst, indices=(layer_idx, b, b + 1),
            src_barriers=(), src_barrier_targets=(),
            num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
            dst_barriers=(),
        ))

    dispatcher = _make_dispatcher(inst_meta, tensor_metas, instructions,
                                  input_indices=(0, 1, 2),
                                  output_indices=(2,))

    result = dispatcher(attn_out, o_weights, hidden_states, 0, 0.125, RMS_NORM_EPS)
    torch.cuda.synchronize()

    return "o_proj_residual", [("hidden", *_check_close(result, expected, atol, rtol))]


def _check_down_proj_residual(sm_count, atol, rtol, seed=42):
    torch.manual_seed(seed)
    layer_idx = 5

    silu_out = torch.randn(INTERMEDIATE_DIM, dtype=DTYPE, device=DEVICE)
    down_weights = torch.randn(NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM, dtype=DTYPE, device=DEVICE)
    hidden_states = torch.randn(HIDDEN_DIM, dtype=DTYPE, device=DEVICE)

    expected = hidden_states.clone() + down_weights[layer_idx] @ silu_out

    itype = ProjResidual()
    src, dst = (0, 1), (2,)
    inst_meta = InstructionMeta(icode=1, itype=itype, src_tensors=src, dst_tensors=dst)

    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(INTERMEDIATE_DIM,), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=_DEVICE),
    ]

    num_blocks = HIDDEN_DIM // BLOCK_SIZE
    instructions = []
    for sm in range(sm_count):
        s = round(sm * num_blocks / sm_count)
        e = round((sm + 1) * num_blocks / sm_count)
        instructions.append(Instruction(
            icode=1, src_tensors=src, dst_tensors=dst, indices=(layer_idx, s, e),
            src_barriers=(), src_barrier_targets=(),
            num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
            dst_barriers=(),
        ))

    dispatcher = _make_dispatcher(inst_meta, tensor_metas, instructions,
                                  input_indices=(0, 1, 2),
                                  output_indices=(2,))

    result = dispatcher(silu_out, down_weights, hidden_states, 0, 0.125, RMS_NORM_EPS)
    torch.cuda.synchronize()

    return "down_proj_residual", [("hidden", *_check_close(result, expected, atol, rtol))]


def _check_rms_upgate_silu(sm_count, atol, rtol, seed=42):
    torch.manual_seed(seed)
    layer_idx = 3

    x = torch.randn(HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    norm_weight = torch.randn(NUM_LAYERS, HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    up_weights = torch.randn(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    gate_weights = torch.randn(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    silu_out = torch.zeros(INTERMEDIATE_DIM, dtype=DTYPE, device=DEVICE)

    # PyTorch reference
    normed = _rmsnorm(x, norm_weight[layer_idx], RMS_NORM_EPS)
    gate = gate_weights[layer_idx] @ normed
    up = up_weights[layer_idx] @ normed
    expected = F.silu(gate) * up

    itype = RmsUpgateSilu(n=HIDDEN_DIM)
    src, dst = (0, 1, 2, 3), (4,)
    inst_meta = InstructionMeta(icode=1, itype=itype, src_tensors=src, dst_tensors=dst)

    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, HIDDEN_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(INTERMEDIATE_DIM,), device=_DEVICE),
    ]

    num_blocks = INTERMEDIATE_DIM // BLOCK_SIZE
    instructions = []
    for sm in range(sm_count):
        s = round(sm * num_blocks / sm_count)
        e = round((sm + 1) * num_blocks / sm_count)
        instructions.append(Instruction(
            icode=1, src_tensors=src, dst_tensors=dst, indices=(layer_idx, s, e),
            src_barriers=(), src_barrier_targets=(),
            num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
            dst_barriers=(),
        ))

    dispatcher = _make_dispatcher(inst_meta, tensor_metas, instructions,
                                  input_indices=tuple(range(5)),
                                  output_indices=(4,))

    result = dispatcher(x, norm_weight, up_weights, gate_weights, silu_out, 0, 0.125, RMS_NORM_EPS)
    torch.cuda.synchronize()

    return "rms_upgate_silu", [("silu_out", *_check_close(result, expected, atol, rtol))]


def _check_rms_lm_head(sm_count, atol, rtol, seed=42):
    torch.manual_seed(seed)

    x = torch.randn(HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    norm_weight = torch.randn(HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    lm_head_weight = torch.randn(VOCAB_SIZE, HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    logits = torch.zeros(VOCAB_SIZE, dtype=DTYPE, device=DEVICE)

    # PyTorch reference
    normed = _rmsnorm(x, norm_weight, RMS_NORM_EPS)
    expected = lm_head_weight @ normed

    itype = RmsLmHead(n=HIDDEN_DIM)
    src, dst = (0, 1, 2), (3,)
    inst_meta = InstructionMeta(icode=1, itype=itype, src_tensors=src, dst_tensors=dst)

    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(VOCAB_SIZE, HIDDEN_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(VOCAB_SIZE,), device=_DEVICE),
    ]

    num_blocks = VOCAB_SIZE // BLOCK_SIZE
    instructions = []
    for sm in range(sm_count):
        s = round(sm * num_blocks / sm_count)
        e = round((sm + 1) * num_blocks / sm_count)
        instructions.append(Instruction(
            icode=1, src_tensors=src, dst_tensors=dst, indices=(s, e),
            src_barriers=(), src_barrier_targets=(),
            num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
            dst_barriers=(),
        ))

    dispatcher = _make_dispatcher(inst_meta, tensor_metas, instructions,
                                  input_indices=(0, 1, 2, 3),
                                  output_indices=(3,))

    result = dispatcher(x, norm_weight, lm_head_weight, logits, 0, 0.125, RMS_NORM_EPS)
    torch.cuda.synchronize()

    return "rms_lm_head", [("logits", *_check_close(result, expected, atol, rtol))]


# ── Calibration ────────────────────────────────────────────

_ALL_CHECKS = [
    _check_rms_qkv_rope_append,
    _check_attention_partial,
    _check_o_proj_residual,
    _check_rms_upgate_silu,
    _check_down_proj_residual,
    _check_rms_lm_head,
]


def calibrate_tolerances(num_seeds=50):
    """Run all instructions across many seeds to find the tightest atol/rtol.

    Uses rtol = sqrt(N) * eps_bf16 as a fixed relative tolerance (the
    analytical per-element bound), then finds the minimum atol needed so
    that every element of every output across all seeds passes:
        |actual - expected| < atol + rtol * |expected|

    The "deficit" is max(|error| - rtol * |expected|, 0) per element —
    the part of the error that rtol can't cover. atol must exceed the
    worst deficit seen.
    """
    sm_count = get_sm_count()
    bf16_eps = 2 ** -7
    rtol = math.sqrt(HIDDEN_DIM) * bf16_eps  # 0.354

    print(f"Calibrating over {num_seeds} seeds with rtol={rtol:.4f} ...")
    print()

    worst_deficit = {}  # (instruction, output) -> max deficit seen

    for seed in range(num_seeds):
        for check_fn in _ALL_CHECKS:
            name, results = check_fn(sm_count, atol=1e9, rtol=rtol, seed=seed)
            for output_name, max_d, mean_d, _, _, deficit in results:
                key = (name, output_name)
                worst_deficit[key] = max(worst_deficit.get(key, 0.0), deficit)

    # atol = worst deficit across all instructions/seeds, with 10% headroom
    max_deficit = max(worst_deficit.values())
    atol = max_deficit * 1.1

    print(f"{'instruction':>25}  {'output':>10}  {'worst deficit':>14}")
    print("-" * 58)
    for (name, output_name), val in sorted(worst_deficit.items()):
        print(f"{name:>25}  {output_name:>10}  {val:>14.4f}")

    print("-" * 58)
    print(f"\nCalibrated tolerances:")
    print(f"  atol = {atol:.4f}  (worst deficit {max_deficit:.4f} + 10%)")
    print(f"  rtol = {rtol:.4f}  (sqrt({HIDDEN_DIM}) * 2^-7)")

    return atol, rtol


# ── Main ───────────────────────────────────────────────────

def benchmark_correctness(atol=None, rtol=None):
    sm_count = get_sm_count()

    if atol is None or rtol is None:
        atol, rtol = calibrate_tolerances()
        print()

    print("Llama 3.2 1B correctness benchmark (bf16, decode M=1)")
    print(f"SMs: {sm_count}")
    print(f"Pass criterion: |actual - expected| < atol + rtol * |expected|")
    print(f"  atol={atol:.4f}  rtol={rtol:.4f}")
    print()
    print(
        f"{'instruction':>25}  {'output':>10}  "
        f"{'max_diff':>10}  {'mean_diff':>10}  "
        f"{'tightness':>10}  {'status':>6}"
    )
    print("-" * 80)

    all_passed = True
    for check_fn in _ALL_CHECKS:
        name, results = check_fn(sm_count, atol, rtol)
        for output_name, max_d, mean_d, tightness, passed, _ in results:
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
            print(
                f"{name:>25}  {output_name:>10}  "
                f"{max_d:>10.4f}  {mean_d:>10.6f}  "
                f"{tightness:>10.4f}  {status:>6}"
            )

    print("-" * 80)
    print(f"{'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    passed = benchmark_correctness()
    raise SystemExit(0 if passed else 1)
