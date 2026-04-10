"""Benchmark Llama 1B decode instructions."""

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
from megakittens.itypes.matvec_adds import MatVecAdds
from megakittens.itypes.rms_lm_head import RmsLmHead
from megakittens.itypes.rms_qkv_rope_append import RmsQkvRopeAppend
from megakittens.itypes.rms_upgate_silu import RmsUpgateSilu
from megakittens.llama1b.scheduler import T, schedule_decode

initialize_cuda_context()

HIDDEN_DIM = 2048
NUM_LAYERS = 16
NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64
GQA_RATIO = NUM_ATTENTION_HEADS // NUM_KV_HEADS  # 4
QKV_DIM = (NUM_ATTENTION_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
Q_DIM = NUM_ATTENTION_HEADS * HEAD_DIM
K_DIM = NUM_KV_HEADS * HEAD_DIM
INTERMEDIATE_DIM = 8192
VOCAB_SIZE = 128256
MAX_SEQ_LEN = 512
BLOCK_SIZE = 16
SEQ_LEN = 128  # decode position for attention benchmark

B300_BW_BYTES_PER_SEC = 8_000_000_000_000

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


def _time_us(fn, warmup=500, iters=1000):
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
    return start.elapsed_time(end) / iters * 1000


def _bench(name, mk_fn, pt_fn, roofline_bytes, warmup=500, iters=1000):
    mk_us = _time_us(mk_fn, warmup, iters)
    pt_us = _time_us(pt_fn, warmup, iters)
    return name, mk_us, pt_us, roofline_bytes


def _print_results(results):
    print(
        f"{'instruction':>25}  {'MK (us)':>10}  {'PyTorch (us)':>13}  "
        f"{'roofline (us)':>14}  {'MK GB/s':>10}  {'PT GB/s':>10}  {'MK/roof':>8}"
    )
    print("-" * 110)
    for name, mk_us, pt_us, total_bytes in results:
        roof_us = total_bytes / B300_BW_BYTES_PER_SEC * 1e6
        mk_gbs = total_bytes / (mk_us * 1e-6) / 1e9
        pt_gbs = total_bytes / (pt_us * 1e-6) / 1e9
        print(
            f"{name:>25}  {mk_us:>10.1f}  {pt_us:>13.1f}  "
            f"{roof_us:>14.2f}  {mk_gbs:>10.1f}  {pt_gbs:>10.1f}  {mk_us / roof_us:>7.1f}x"
        )


# ── Unfused PyTorch references ──────────────────────────────

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


def _pt_rms_qkv_rope_append(hidden, norm_w, qkv_w, rope_cos, rope_sin, k_buf, v_buf):
    h = torch.rms_norm(hidden, [hidden.shape[-1]], norm_w, 1e-5)
    qkv = qkv_w @ h
    q = qkv[:Q_DIM].view(NUM_ATTENTION_HEADS, HEAD_DIM)
    k = qkv[Q_DIM:Q_DIM + K_DIM].view(NUM_KV_HEADS, HEAD_DIM)
    v = qkv[Q_DIM + K_DIM:].view(NUM_KV_HEADS, HEAD_DIM)
    cos = rope_cos[0].bfloat16()
    sin = rope_sin[0].bfloat16()
    q = _apply_rope(q, cos, sin)
    k = _apply_rope(k, cos, sin)
    k_buf[0] = k
    v_buf[0] = v
    return q.reshape(-1)


def _pt_attention_partial(q, k_cache, v_cache, layer_idx, seq_len, attn_scale):
    q_heads = q.view(NUM_KV_HEADS, GQA_RATIO, HEAD_DIM)  # [8, 4, 64]
    k = k_cache[layer_idx, :seq_len].permute(1, 2, 0)     # [8, 64, seq_len]
    v = v_cache[layer_idx, :seq_len].permute(1, 0, 2)     # [8, seq_len, 64]
    scores = torch.bmm(q_heads, k) * attn_scale           # [8, 4, seq_len]
    w = F.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.bmm(w, v)                                 # [8, 4, 64]
    return out.reshape(-1)


def _pt_proj_residual(input_vec, weights, residual):
    return residual + weights @ input_vec


def _pt_rms_upgate_silu(x, norm_weight, up_weights, gate_weights):
    normed = _rmsnorm(x, norm_weight, 1e-5)
    gate = gate_weights @ normed
    up = up_weights @ normed
    return F.silu(gate) * up


def _pt_rms_lm_head(x, norm_weight, lm_head_weight):
    normed = _rmsnorm(x, norm_weight, 1e-5)
    return lm_head_weight @ normed


# ── Instruction setups ──────────────────────────────────────
# Each returns (name, mk_fn, pt_fn, roofline_bytes).

def _setup_rms_qkv_rope_append(sm_count):
    itype = RmsQkvRopeAppend(n=HIDDEN_DIM, head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS)
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
            icode=1, src_tensors=src, dst_tensors=dst, indices=(0, s, e),
            src_barriers=(), src_barrier_targets=(),
            num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
            dst_barriers=(),
        ))

    dispatcher = _make_dispatcher(inst_meta, tensor_metas, instructions,
                                  input_indices=tuple(range(8)),
                                  output_indices=(7, 5, 6))

    D = "cuda"
    tensors = [
        torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(NUM_LAYERS, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(NUM_LAYERS, QKV_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=D),
        torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=D),
        torch.zeros(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D),
        torch.zeros(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D),
        torch.zeros(Q_DIM, dtype=torch.bfloat16, device=D),
    ]

    mk_fn = lambda: dispatcher(*tensors, 0, 0.125, 1e-5)

    hidden, norm_w, qkv_w = tensors[0], tensors[1][0], tensors[2][0]
    cos, sin = tensors[3], tensors[4]
    k_buf = torch.zeros(MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    v_buf = torch.zeros(MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    pt_fn = lambda: _pt_rms_qkv_rope_append(hidden, norm_w, qkv_w, cos, sin, k_buf, v_buf)

    roofline_bytes = (HIDDEN_DIM + HIDDEN_DIM) * 2 + QKV_DIM * HIDDEN_DIM * 2 \
                   + 2 * HEAD_DIM * 4 + (Q_DIM + K_DIM + K_DIM) * 2

    return "rms_qkv_rope_append", mk_fn, pt_fn, roofline_bytes


def _setup_attention_partial(sm_count):
    itype = AttentionPartial()
    src, dst = (0, 1, 2), (3,)
    inst_meta = InstructionMeta(icode=1, itype=itype, src_tensors=src, dst_tensors=dst)

    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(Q_DIM,), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(Q_DIM,), device=_DEVICE),
    ]

    layer_idx = 0
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

    D = "cuda"
    q = torch.randn(Q_DIM, dtype=torch.bfloat16, device=D)
    k_cache = torch.randn(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    v_cache = torch.randn(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    attn_out = torch.zeros(Q_DIM, dtype=torch.bfloat16, device=D)

    pos_id = SEQ_LEN - 1
    attn_scale = 1.0 / math.sqrt(HEAD_DIM)
    mk_fn = lambda: dispatcher(q, k_cache, v_cache, attn_out, pos_id, attn_scale, 1e-5)
    pt_fn = lambda: _pt_attention_partial(q, k_cache, v_cache, layer_idx, SEQ_LEN, attn_scale)

    roofline_bytes = Q_DIM * 2 \
                   + 2 * SEQ_LEN * NUM_KV_HEADS * HEAD_DIM * 2 \
                   + Q_DIM * 2

    return "attention_partial", mk_fn, pt_fn, roofline_bytes


def _setup_o_proj_residual(sm_count):
    itype = MatVecAdds(n=HIDDEN_DIM)
    src, dst = (0, 1), (2,)
    inst_meta = InstructionMeta(icode=1, itype=itype, src_tensors=src, dst_tensors=dst)

    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=_DEVICE),
    ]

    layer_idx = 0
    num_blocks = HIDDEN_DIM // BLOCK_SIZE  # 128
    instructions = []
    for b in range(num_blocks):
        instructions.append(Instruction(
            icode=1, src_tensors=src, dst_tensors=dst, indices=(layer_idx, b, b + 1, 0),
            src_barriers=(), src_barrier_targets=(),
            num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
            dst_barriers=(),
        ))

    dispatcher = _make_dispatcher(inst_meta, tensor_metas, instructions,
                                  input_indices=(0, 1, 2),
                                  output_indices=(2,))

    D = "cuda"
    attn_out_vec = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    o_weights = torch.randn(NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    hidden_states = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=D)

    mk_fn = lambda: dispatcher(attn_out_vec, o_weights, hidden_states, 0, 0.125, 1e-5)
    pt_fn = lambda: _pt_proj_residual(attn_out_vec, o_weights[layer_idx], hidden_states)

    roofline_bytes = HIDDEN_DIM * 2 + HIDDEN_DIM * HIDDEN_DIM * 2 + HIDDEN_DIM * 2

    return "o_proj_residual", mk_fn, pt_fn, roofline_bytes


def _setup_rms_upgate_silu(sm_count):
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

    layer_idx = 0
    num_blocks = INTERMEDIATE_DIM // BLOCK_SIZE
    instructions = []
    for sm in range(sm_count):
        instructions.append(Instruction(
            icode=1, src_tensors=src, dst_tensors=dst,
            indices=(layer_idx, sm, sm_count, num_blocks, 0),
            src_barriers=(), src_barrier_targets=(),
            num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
            dst_barriers=(),
        ))

    dispatcher = _make_dispatcher(inst_meta, tensor_metas, instructions,
                                  input_indices=tuple(range(5)),
                                  output_indices=(4,))

    D = "cuda"
    tensors = [
        torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(NUM_LAYERS, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.zeros(INTERMEDIATE_DIM, dtype=torch.bfloat16, device=D),
    ]

    mk_fn = lambda: dispatcher(*tensors, 0, 0.125, 1e-5)

    x, norm_w = tensors[0], tensors[1][layer_idx]
    up_w, gate_w = tensors[2][layer_idx], tensors[3][layer_idx]
    pt_fn = lambda: _pt_rms_upgate_silu(x, norm_w, up_w, gate_w)

    roofline_bytes = (HIDDEN_DIM + HIDDEN_DIM) * 2 \
                   + 2 * INTERMEDIATE_DIM * HIDDEN_DIM * 2 \
                   + INTERMEDIATE_DIM * 2

    return "rms_upgate_silu", mk_fn, pt_fn, roofline_bytes


def _setup_down_proj_residual(sm_count):
    N = HIDDEN_DIM  # pipeline reduction capacity
    num_chunks = INTERMEDIATE_DIM // N  # 4
    itype = MatVecAdds(n=N)
    src, dst = (0, 1), (2,)
    inst_meta = InstructionMeta(icode=1, itype=itype, src_tensors=src, dst_tensors=dst)

    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(INTERMEDIATE_DIM,), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=_DEVICE),
    ]

    layer_idx = 0
    num_blocks = HIDDEN_DIM // BLOCK_SIZE
    instructions = []
    for chunk in range(num_chunks):
        reduction_col_offset = chunk * N
        for block in range(num_blocks):
            instructions.append(Instruction(
                icode=1, src_tensors=src, dst_tensors=dst,
                indices=(layer_idx, block, block + 1, reduction_col_offset),
                src_barriers=(), src_barrier_targets=(),
                num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
                dst_barriers=(),
            ))

    dispatcher = _make_dispatcher(inst_meta, tensor_metas, instructions,
                                  input_indices=(0, 1, 2),
                                  output_indices=(2,))

    D = "cuda"
    silu_out = torch.randn(INTERMEDIATE_DIM, dtype=torch.bfloat16, device=D)
    down_weights = torch.randn(NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM, dtype=torch.bfloat16, device=D)
    hidden_states = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=D)

    mk_fn = lambda: dispatcher(silu_out, down_weights, hidden_states, 0, 0.125, 1e-5)
    pt_fn = lambda: _pt_proj_residual(silu_out, down_weights[layer_idx], hidden_states)

    roofline_bytes = INTERMEDIATE_DIM * 2 + HIDDEN_DIM * INTERMEDIATE_DIM * 2 + HIDDEN_DIM * 2

    return "down_proj_residual", mk_fn, pt_fn, roofline_bytes


def _setup_rms_lm_head(sm_count):
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
    if VOCAB_SIZE % BLOCK_SIZE != 0:
        num_blocks += 1
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

    D = "cuda"
    x = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    norm_weight = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    lm_head_weight = torch.randn(VOCAB_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    logits = torch.zeros(VOCAB_SIZE, dtype=torch.bfloat16, device=D)

    mk_fn = lambda: dispatcher(x, norm_weight, lm_head_weight, logits, 0, 0.125, 1e-5)
    pt_fn = lambda: _pt_rms_lm_head(x, norm_weight, lm_head_weight)

    roofline_bytes = (HIDDEN_DIM + HIDDEN_DIM) * 2 + VOCAB_SIZE * HIDDEN_DIM * 2 + VOCAB_SIZE * 2

    return "rms_lm_head", mk_fn, pt_fn, roofline_bytes


def benchmark_instructions():
    sm_count = get_sm_count()
    print("Llama 3.2 1B instruction benchmarks (bf16, decode M=1)")
    print(f"B300 theoretical peak bandwidth: {B300_BW_BYTES_PER_SEC / 1e12:.0f} TB/s")
    print(f"Attention seq_len: {SEQ_LEN}")
    print()

    results = [
        _bench(*_setup_rms_qkv_rope_append(sm_count)),
        _bench(*_setup_attention_partial(sm_count)),
        _bench(*_setup_o_proj_residual(sm_count)),
        _bench(*_setup_rms_upgate_silu(sm_count)),
        _bench(*_setup_down_proj_residual(sm_count)),
        _bench(*_setup_rms_lm_head(sm_count)),
    ]
    _print_results(results)


# ── Full decode benchmark ──────────────────────────────────

def _pt_decode(hidden, weights, k_cache, v_cache, rope_cos, rope_sin, pos_id):
    seq_len = pos_id + 1
    attn_scale = 1.0 / math.sqrt(HEAD_DIM)
    cos = rope_cos[pos_id].bfloat16()
    sin = rope_sin[pos_id].bfloat16()

    x = hidden.clone()
    for layer in range(NUM_LAYERS):
        normed = _rmsnorm(x, weights["attn_norm_weights"][layer], 1e-5)
        qkv = weights["qkv_weights"][layer] @ normed
        q = qkv[:Q_DIM].view(NUM_ATTENTION_HEADS, HEAD_DIM)
        k = qkv[Q_DIM:Q_DIM + K_DIM].view(NUM_KV_HEADS, HEAD_DIM)
        v = qkv[Q_DIM + K_DIM:].view(NUM_KV_HEADS, HEAD_DIM)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        k_cache[layer, pos_id] = k
        v_cache[layer, pos_id] = v

        q_grouped = q.view(NUM_KV_HEADS, GQA_RATIO, HEAD_DIM)
        k_cached = k_cache[layer, :seq_len].permute(1, 2, 0)
        v_cached = v_cache[layer, :seq_len].permute(1, 0, 2)
        scores = torch.bmm(q_grouped, k_cached) * attn_scale
        w = F.softmax(scores.float(), dim=-1).to(torch.bfloat16)
        attn_out = torch.bmm(w, v_cached).reshape(-1)

        x = x + weights["o_weights"][layer] @ attn_out

        normed = _rmsnorm(x, weights["mlp_norm_weights"][layer], 1e-5)
        gate = weights["gate_weights"][layer] @ normed
        up = weights["up_weights"][layer] @ normed
        silu_out = F.silu(gate) * up

        x = x + weights["down_weights"][layer] @ silu_out

    normed = _rmsnorm(x, weights["lm_head_norm_weight"], 1e-5)
    return weights["lm_head_weight"] @ normed


def benchmark_decode_e2e():
    sm_count = get_sm_count()
    pos_id = SEQ_LEN - 1
    attn_scale = 1.0 / math.sqrt(HEAD_DIM)

    schedule = schedule_decode(sm_count=sm_count)
    instruction_metas, tensor_metas, instructions, num_barriers, input_indices, output_indices = schedule

    dispatcher = Dispatcher(
        instruction_metas, tensor_metas, instructions, num_barriers,
        input_indices, output_indices,
        use_jit_cache=False,
        scalar_fields=SCALARS,
    )

    D = "cuda"
    weights = {
        "qkv_weights": torch.randn(NUM_LAYERS, QKV_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        "o_weights": torch.randn(NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        "attn_norm_weights": torch.randn(NUM_LAYERS, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        "mlp_norm_weights": torch.randn(NUM_LAYERS, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        "up_weights": torch.randn(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        "gate_weights": torch.randn(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        "down_weights": torch.randn(NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM, dtype=torch.bfloat16, device=D),
        "lm_head_norm_weight": torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        "lm_head_weight": torch.randn(VOCAB_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
    }

    hidden = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    mk_k_cache = torch.randn(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    mk_v_cache = torch.randn(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    rope_cos = torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=D)
    rope_sin = torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=D)

    mk_tensors = [
        weights["qkv_weights"],
        weights["o_weights"],
        weights["attn_norm_weights"],
        weights["mlp_norm_weights"],
        weights["up_weights"],
        weights["gate_weights"],
        weights["down_weights"],
        weights["lm_head_norm_weight"],
        weights["lm_head_weight"],
        hidden,
        torch.zeros(Q_DIM, dtype=torch.bfloat16, device=D),
        torch.zeros(HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.zeros(INTERMEDIATE_DIM, dtype=torch.bfloat16, device=D),
        torch.zeros(VOCAB_SIZE, dtype=torch.bfloat16, device=D),
        mk_k_cache,
        mk_v_cache,
        rope_cos,
        rope_sin,
    ]

    mk_fn = lambda: dispatcher(*mk_tensors, pos_id, attn_scale, 1e-5)

    pt_k_cache = mk_k_cache.clone()
    pt_v_cache = mk_v_cache.clone()
    pt_fn = lambda: _pt_decode(hidden, weights, pt_k_cache, pt_v_cache, rope_cos, rope_sin, pos_id)

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
    roofline_bytes = NUM_LAYERS * per_layer_bytes + lm_head_bytes
    roof_us = roofline_bytes / B300_BW_BYTES_PER_SEC * 1e6

    mk_us = _time_us(mk_fn, warmup=100, iters=500)
    pt_us = _time_us(pt_fn, warmup=100, iters=500)
    mk_gbs = roofline_bytes / (mk_us * 1e-6) / 1e9
    pt_gbs = roofline_bytes / (pt_us * 1e-6) / 1e9

    print()
    print("Llama 3.2 1B full decode (bf16, M=1)")
    print(f"Layers: {NUM_LAYERS}, seq_len: {SEQ_LEN}")
    print(f"Total model bytes: {roofline_bytes / 1e9:.2f} GB")
    print()
    print(f"  Megakernel:  {mk_us:>10.1f} us  ({mk_gbs:.1f} GB/s)  {mk_us / roof_us:.1f}x roofline")
    print(f"  PyTorch:     {pt_us:>10.1f} us  ({pt_gbs:.1f} GB/s)  {pt_us / roof_us:.1f}x roofline")
    print(f"  Roofline:    {roof_us:>10.1f} us")



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
    attn_scale = 1.0 / math.sqrt(HEAD_DIM)
    for pos in range(len(token_ids)):
        x = weights["embed_weight"][token_ids[pos]]
        cos = rope_cos[pos]
        sin = rope_sin[pos]
        seq_len = pos + 1

        for layer_idx in range(NUM_LAYERS):
            normed = _rmsnorm(x, weights["attn_norm_weights"][layer_idx], 1e-5)
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
                    scores = (q[q_head] @ k_cached.T) * attn_scale
                    if seq_len > 1:
                        mask = torch.full((seq_len,), float("-inf"), device=x.device)
                        mask[:pos + 1] = 0.0
                        scores = scores + mask
                    w = F.softmax(scores.float(), dim=-1).to(x.dtype)
                    attn_out[q_head] = w @ v_cached

            x = x + weights["o_weights"][layer_idx] @ attn_out.reshape(HIDDEN_DIM)

            normed_mlp = _rmsnorm(x, weights["mlp_norm_weights"][layer_idx], 1e-5)
            gate = weights["gate_weights"][layer_idx] @ normed_mlp
            up = weights["up_weights"][layer_idx] @ normed_mlp
            x = x + weights["down_weights"][layer_idx] @ (F.silu(gate) * up)

    return x


def benchmark_tok_per_sec(prompt="Hello, my name is", max_new_tokens=200, num_samples=5, warmup=5):
    """Benchmark decode throughput (tokens/sec) with real HF weights.

    Loads Llama-3.2-1B-Instruct from HuggingFace, prefills KV cache with
    prompt, then times decode steps. Matches gpt-fast --greedy methodology.
    """
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sm_count = get_sm_count()
    attn_scale = 1.0 / math.sqrt(HEAD_DIM)
    D = "cuda"

    # Load real weights from HuggingFace
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

    del hf_model  # free HF model memory

    # Prefill KV cache with prompt tokens
    k_cache = torch.zeros(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    v_cache = torch.zeros(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    print("Prefilling KV cache...")
    with torch.inference_mode():
        last_hidden = _prefill_kv_cache(input_ids, weights, k_cache, v_cache, rope_cos, rope_sin)
    # First decode token comes from prefill logits
    prefill_logits = weights["lm_head_weight"] @ _rmsnorm(last_hidden, weights["lm_head_norm_weight"], 1e-5)
    first_token = torch.argmax(prefill_logits)
    # Snapshot prefilled cache for reset between runs
    k_cache_snapshot = k_cache.clone()
    v_cache_snapshot = v_cache.clone()

    # Build megakernel dispatcher
    schedule = schedule_decode(sm_count=sm_count)
    instruction_metas, tensor_metas, instructions, num_barriers, input_indices, output_indices = schedule

    dispatcher = Dispatcher(
        instruction_metas, tensor_metas, instructions, num_barriers,
        input_indices, output_indices,
        use_jit_cache=False,
        scalar_fields=SCALARS,
    )

    # Assemble tensors in scheduler T.* order using real weights
    hidden_states = embed_weight[first_token].clone()
    logits = torch.zeros(VOCAB_SIZE, dtype=torch.bfloat16, device=D)

    mk_tensors = [
        weights["qkv_weights"],         # T.QKV_WEIGHTS
        weights["o_weights"],           # T.O_WEIGHTS
        weights["attn_norm_weights"],   # T.ATTN_NORM_WEIGHTS
        weights["mlp_norm_weights"],    # T.MLP_NORM_WEIGHTS
        weights["up_weights"],          # T.UP_WEIGHTS
        weights["gate_weights"],        # T.GATE_WEIGHTS
        weights["down_weights"],        # T.DOWN_WEIGHTS
        weights["lm_head_norm_weight"], # T.LM_HEAD_NORM_WEIGHT
        weights["lm_head_weight"],      # T.LM_HEAD_WEIGHT
        hidden_states,                  # T.HIDDEN_STATES
        torch.zeros(Q_DIM, dtype=torch.bfloat16, device=D),           # T.Q_POST_ROPE
        torch.zeros(HIDDEN_DIM, dtype=torch.bfloat16, device=D),      # T.ATTN_OUT
        torch.zeros(INTERMEDIATE_DIM, dtype=torch.bfloat16, device=D), # T.SILU_OUT
        logits,                         # T.LOGITS
        k_cache,                        # T.K_CACHE
        v_cache,                        # T.V_CACHE
        rope_cos,                       # T.ROPE_COS
        rope_sin,                       # T.ROPE_SIN
    ]

    # Model size: non-embedding params only (matches gpt-fast _get_model_size)
    model_size = sum(t.nelement() * t.element_size() for t in mk_tensors[:9])
    params = sum(t.nelement() for t in mk_tensors[:9])

    embedding = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_DIM, device=D, dtype=torch.bfloat16)
    embedding.weight.data.copy_(embed_weight)
    num_decode_tokens = max_new_tokens - 1  # 199, first token comes from prefill
    output_tokens = torch.zeros(max_new_tokens, dtype=torch.long, device=D)

    def _decode_step(pos_id, input_token):
        hidden_states.copy_(embedding(input_token))
        dispatcher.relaunch(pos_id=pos_id)
        return torch.argmax(logits, dim=-1)

    # Warmup: first call via call() to build cache, then relaunch
    dispatcher(*mk_tensors, prompt_len, attn_scale, 1e-5)
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

    # Timed runs
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

    # Print generated text from last run
    all_ids = torch.cat([input_ids.to(D), output_tokens[:max_new_tokens]])
    print(tokenizer.decode(all_ids.tolist()))
    print()

    print("==========")
    print(f"Prompt Length: {prompt_len}")
    print(f"Generated tokens: {max_new_tokens}")
    print(f"Average tokens/sec (decode only): {torch.mean(torch.tensor(decode_tokens_per_sec_list)).item():.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == "__main__":
    benchmark_instructions()
    benchmark_decode_e2e()
    benchmark_tok_per_sec()
