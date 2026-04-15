"""Benchmark Llama 1B decode instructions."""

import ctypes
import math

import cuda.bindings.driver as cuda_driver
import torch
import torch.nn.functional as F

from megakittens.jit.cuda_utils import get_sm_count, initialize_cuda_context
from megakittens.dispatcher import Dispatcher
from megakittens.schema.device import Device
from megakittens.schema.dtype import DType
from megakittens.schema.instruction import Instruction, InstructionMeta
from megakittens.schema.tensor import TensorMeta
from megakittens.itypes.noop import Noop
from megakittens.itypes.llama1b.attention_partial import AttentionPartial
from megakittens.itypes.llama1b.matvec_adds import MatVecAdds
from megakittens.itypes.llama1b.rms_lm_head import RmsLmHead
from megakittens.itypes.llama1b.rms_qkv_rope_append import RmsQkvRopeAppend
from megakittens.itypes.llama1b.rms_upgate_silu import RmsUpgateSilu
from .scheduler import (
    GQA_RATIO,
    HEAD_DIM,
    HIDDEN_DIM,
    INTERMEDIATE_DIM,
    K_DIM,
    MATVEC_BLOCK_SIZE,
    MAX_SEQ_LEN,
    NUM_ATTENTION_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    Q_DIM,
    QKV_DIM,
    RMS_NORM_EPS,
    VOCAB_SIZE,
    schedule_decode,
)

initialize_cuda_context()

SEQ_LEN = 128  # decode position for attention benchmark

B300_BW_BYTES_PER_SEC = 8_000_000_000_000
ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)

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


# --- pytorch refs


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
    h = torch.rms_norm(hidden, [hidden.shape[-1]], norm_w, RMS_NORM_EPS)
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
    qh = q.view(NUM_KV_HEADS, GQA_RATIO, HEAD_DIM)
    k = k_cache[layer_idx, :seq_len].permute(1, 2, 0)
    v = v_cache[layer_idx, :seq_len].permute(1, 0, 2)
    scores = torch.bmm(qh, k) * attn_scale
    w = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.bmm(w, v).reshape(-1)


def _pt_proj_residual(input_vec, weights, residual):
    return residual + weights @ input_vec


def _pt_rms_upgate_silu(x, norm_weight, up_weights, gate_weights):
    normed = _rmsnorm(x, norm_weight, RMS_NORM_EPS)
    gate = gate_weights @ normed
    up = up_weights @ normed
    return F.silu(gate) * up


def _pt_rms_lm_head(x, norm_weight, lm_head_weight):
    normed = _rmsnorm(x, norm_weight, RMS_NORM_EPS)
    return lm_head_weight @ normed


# --- per-instruction dispatchers + roofline byte counts


def _setup_rms_qkv_rope_append(sm_count):
    itype = RmsQkvRopeAppend(n=HIDDEN_DIM, head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS)
    src, dst = (0, 1, 2, 3, 4, 5, 6, 8, 9), (7,)
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
        TensorMeta(dtype=DType.int32, shape=(1,), device=_DEVICE),   # pos_id
        TensorMeta(dtype=DType.fp32, shape=(1,), device=_DEVICE),    # rms_norm_eps
    ]

    num_blocks = QKV_DIM // MATVEC_BLOCK_SIZE
    instructions = []
    for sm in range(sm_count):
        s = round(sm * num_blocks / sm_count)
        e = round((sm + 1) * num_blocks / sm_count)
        instructions.append(Instruction(
            icode=1, src_tensors=(0, 1, 2, 3, 4, 5, 6), dst_tensors=(7,), indices=(0, s, e),
            src_barriers=(), src_barrier_targets=(),
            num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
            dst_barriers=(),
        ))

    dispatcher = _make_dispatcher(inst_meta, tensor_metas, instructions,
                                  input_indices=tuple(range(10)),
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
        torch.tensor([0], dtype=torch.int32, device=D),             # pos_id
        torch.tensor([RMS_NORM_EPS], dtype=torch.float32, device=D), # rms_norm_eps
    ]

    mk_fn = lambda: dispatcher(*tensors)

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
    src, dst = (0, 1, 2, 4, 5), (3,)
    inst_meta = InstructionMeta(icode=1, itype=itype, src_tensors=src, dst_tensors=dst)

    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(Q_DIM,), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(Q_DIM,), device=_DEVICE),
        TensorMeta(dtype=DType.int32, shape=(1,), device=_DEVICE),   # pos_id
        TensorMeta(dtype=DType.fp32, shape=(1,), device=_DEVICE),    # attn_scale
    ]

    layer_idx = 0
    instructions = []
    for kv_head in range(NUM_KV_HEADS):
        instructions.append(Instruction(
            icode=1, src_tensors=(0, 1, 2), dst_tensors=(3,), indices=(layer_idx, kv_head),
            src_barriers=(), src_barrier_targets=(),
            num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
            dst_barriers=(),
        ))

    dispatcher = _make_dispatcher(inst_meta, tensor_metas, instructions,
                                  input_indices=tuple(range(6)),
                                  output_indices=(3,))

    D = "cuda"
    q = torch.randn(Q_DIM, dtype=torch.bfloat16, device=D)
    k_cache = torch.randn(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    v_cache = torch.randn(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    attn_out = torch.zeros(Q_DIM, dtype=torch.bfloat16, device=D)
    pos_id_t = torch.tensor([SEQ_LEN - 1], dtype=torch.int32, device=D)
    attn_scale_t = torch.tensor([ATTN_SCALE], dtype=torch.float32, device=D)

    mk_fn = lambda: dispatcher(q, k_cache, v_cache, attn_out, pos_id_t, attn_scale_t)
    pt_fn = lambda: _pt_attention_partial(q, k_cache, v_cache, layer_idx, SEQ_LEN, ATTN_SCALE)

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
    num_blocks = HIDDEN_DIM // MATVEC_BLOCK_SIZE
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

    mk_fn = lambda: dispatcher(attn_out_vec, o_weights, hidden_states)
    pt_fn = lambda: _pt_proj_residual(attn_out_vec, o_weights[layer_idx], hidden_states)

    roofline_bytes = HIDDEN_DIM * 2 + HIDDEN_DIM * HIDDEN_DIM * 2 + HIDDEN_DIM * 2

    return "o_proj_residual", mk_fn, pt_fn, roofline_bytes


def _setup_rms_upgate_silu(sm_count):
    itype = RmsUpgateSilu(n=HIDDEN_DIM)
    src, dst = (0, 1, 2, 3, 5), (4,)
    inst_meta = InstructionMeta(icode=1, itype=itype, src_tensors=src, dst_tensors=dst)

    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, HIDDEN_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(INTERMEDIATE_DIM,), device=_DEVICE),
        TensorMeta(dtype=DType.fp32, shape=(1,), device=_DEVICE),    # rms_norm_eps
    ]

    layer_idx = 0
    num_blocks = INTERMEDIATE_DIM // MATVEC_BLOCK_SIZE
    instructions = []
    for sm in range(sm_count):
        instructions.append(Instruction(
            icode=1, src_tensors=(0, 1, 2, 3), dst_tensors=(4,),
            indices=(layer_idx, sm, sm_count, num_blocks, 0),
            src_barriers=(), src_barrier_targets=(),
            num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
            dst_barriers=(),
        ))

    dispatcher = _make_dispatcher(inst_meta, tensor_metas, instructions,
                                  input_indices=tuple(range(6)),
                                  output_indices=(4,))

    D = "cuda"
    tensors = [
        torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(NUM_LAYERS, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.zeros(INTERMEDIATE_DIM, dtype=torch.bfloat16, device=D),
        torch.tensor([RMS_NORM_EPS], dtype=torch.float32, device=D), # rms_norm_eps
    ]

    mk_fn = lambda: dispatcher(*tensors)

    x, norm_w = tensors[0], tensors[1][layer_idx]
    up_w, gate_w = tensors[2][layer_idx], tensors[3][layer_idx]
    pt_fn = lambda: _pt_rms_upgate_silu(x, norm_w, up_w, gate_w)

    roofline_bytes = (HIDDEN_DIM + HIDDEN_DIM) * 2 \
                   + 2 * INTERMEDIATE_DIM * HIDDEN_DIM * 2 \
                   + INTERMEDIATE_DIM * 2

    return "rms_upgate_silu", mk_fn, pt_fn, roofline_bytes


def _setup_down_proj_residual(sm_count):
    N = HIDDEN_DIM
    num_chunks = INTERMEDIATE_DIM // N
    itype = MatVecAdds(n=N)
    src, dst = (0, 1), (2,)
    inst_meta = InstructionMeta(icode=1, itype=itype, src_tensors=src, dst_tensors=dst)

    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(INTERMEDIATE_DIM,), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=_DEVICE),
    ]

    layer_idx = 0
    num_blocks = HIDDEN_DIM // MATVEC_BLOCK_SIZE
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

    mk_fn = lambda: dispatcher(silu_out, down_weights, hidden_states)
    pt_fn = lambda: _pt_proj_residual(silu_out, down_weights[layer_idx], hidden_states)

    roofline_bytes = INTERMEDIATE_DIM * 2 + HIDDEN_DIM * INTERMEDIATE_DIM * 2 + HIDDEN_DIM * 2

    return "down_proj_residual", mk_fn, pt_fn, roofline_bytes


def _setup_rms_lm_head(sm_count):
    itype = RmsLmHead(n=HIDDEN_DIM)
    src, dst = (0, 1, 2, 4), (3,)
    inst_meta = InstructionMeta(icode=1, itype=itype, src_tensors=src, dst_tensors=dst)

    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(VOCAB_SIZE, HIDDEN_DIM), device=_DEVICE),
        TensorMeta(dtype=DType.bf16, shape=(VOCAB_SIZE,), device=_DEVICE),
        TensorMeta(dtype=DType.fp32, shape=(1,), device=_DEVICE),    # rms_norm_eps
    ]

    num_blocks = VOCAB_SIZE // MATVEC_BLOCK_SIZE
    if VOCAB_SIZE % MATVEC_BLOCK_SIZE != 0:
        num_blocks += 1
    instructions = []
    for sm in range(sm_count):
        s = round(sm * num_blocks / sm_count)
        e = round((sm + 1) * num_blocks / sm_count)
        instructions.append(Instruction(
            icode=1, src_tensors=(0, 1, 2), dst_tensors=(3,), indices=(s, e),
            src_barriers=(), src_barrier_targets=(),
            num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
            dst_barriers=(),
        ))

    dispatcher = _make_dispatcher(inst_meta, tensor_metas, instructions,
                                  input_indices=tuple(range(5)),
                                  output_indices=(3,))

    D = "cuda"
    x = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    norm_weight = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    lm_head_weight = torch.randn(VOCAB_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    logits = torch.zeros(VOCAB_SIZE, dtype=torch.bfloat16, device=D)
    rms_norm_eps_t = torch.tensor([RMS_NORM_EPS], dtype=torch.float32, device=D)

    mk_fn = lambda: dispatcher(x, norm_weight, lm_head_weight, logits, rms_norm_eps_t)
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


# --- full model step (random weights) vs pytorch


def _pt_decode(hidden, weights, k_cache, v_cache, rope_cos, rope_sin, pos_id):
    seq_len = pos_id + 1
    cos = rope_cos[pos_id].bfloat16()
    sin = rope_sin[pos_id].bfloat16()

    x = hidden.clone()
    for layer in range(NUM_LAYERS):
        normed = _rmsnorm(x, weights["attn_norm_weights"][layer], RMS_NORM_EPS)
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
        scores = torch.bmm(q_grouped, k_cached) * ATTN_SCALE
        w = F.softmax(scores.float(), dim=-1).to(torch.bfloat16)
        attn_out = torch.bmm(w, v_cached).reshape(-1)

        x = x + weights["o_weights"][layer] @ attn_out

        normed = _rmsnorm(x, weights["mlp_norm_weights"][layer], RMS_NORM_EPS)
        gate = weights["gate_weights"][layer] @ normed
        up = weights["up_weights"][layer] @ normed
        silu_out = F.silu(gate) * up

        x = x + weights["down_weights"][layer] @ silu_out

    normed = _rmsnorm(x, weights["lm_head_norm_weight"], RMS_NORM_EPS)
    return weights["lm_head_weight"] @ normed


def benchmark_decode_e2e(num_partitions: int = 1):
    sm_count = get_sm_count()
    pos_id = 4096 - 1
    seq_len = pos_id + 1

    schedule = schedule_decode(sm_count=sm_count, num_partitions=num_partitions)
    instruction_metas, tensor_metas, instructions, num_barriers, input_indices, output_indices = schedule

    dispatcher = Dispatcher(
        instruction_metas, tensor_metas, instructions, num_barriers,
        input_indices, output_indices,
        use_jit_cache=False,
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
    pos_id_tensor = torch.tensor([pos_id], dtype=torch.int32, device=D)
    attn_scale_tensor = torch.tensor([ATTN_SCALE], dtype=torch.float32, device=D)
    rms_norm_eps_tensor = torch.tensor([RMS_NORM_EPS], dtype=torch.float32, device=D)

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
        pos_id_tensor,
        attn_scale_tensor,
        rms_norm_eps_tensor,
        torch.zeros(NUM_ATTENTION_HEADS, num_partitions, HEAD_DIM, dtype=torch.float32, device=D),
        torch.zeros(NUM_ATTENTION_HEADS, ((num_partitions + 15) // 16) * 16, dtype=torch.float32, device=D),
    ]

    mk_fn = lambda: dispatcher(*mk_tensors)

    pt_k_cache = mk_k_cache.clone()
    pt_v_cache = mk_v_cache.clone()
    pt_fn = lambda: _pt_decode(hidden, weights, pt_k_cache, pt_v_cache, rope_cos, rope_sin, pos_id)

    per_layer_bytes = (
        QKV_DIM * HIDDEN_DIM * 2
        + HIDDEN_DIM * 2
        + 2 * HEAD_DIM * 4
        + 2 * seq_len * NUM_KV_HEADS * HEAD_DIM * 2
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

    label = f"num_partitions={num_partitions}" if num_partitions > 1 else "no reduction"
    print()
    print(f"Llama 3.2 1B full decode (bf16, M=1, {label})")
    print(f"Layers: {NUM_LAYERS}, seq_len: {seq_len}")
    print(f"Total model bytes: {roofline_bytes / 1e9:.2f} GB")
    print()
    print(f"  Megakernel:  {mk_us:>10.1f} us  ({mk_gbs:.1f} GB/s)  {mk_us / roof_us:.1f}x roofline")
    print(f"  PyTorch:     {pt_us:>10.1f} us  ({pt_gbs:.1f} GB/s)  {pt_us / roof_us:.1f}x roofline")
    print(f"  Roofline:    {roof_us:>10.1f} us")


def check_reduction_correctness(num_partitions: int = 16):
    """Compare single-partition vs multi-partition megakernel on same inputs."""
    sm_count = get_sm_count()
    pos_id = 4096 - 1
    D = "cuda"

    # Shared inputs
    hidden = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=D)
    weights_list = [
        torch.randn(NUM_LAYERS, QKV_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(NUM_LAYERS, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(NUM_LAYERS, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(VOCAB_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
    ]
    k_cache = torch.randn(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    v_cache = torch.randn(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    rope_cos = torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=D)
    rope_sin = torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=D)
    pos_id_t = torch.tensor([pos_id], dtype=torch.int32, device=D)
    scale_t = torch.tensor([ATTN_SCALE], dtype=torch.float32, device=D)
    eps_t = torch.tensor([RMS_NORM_EPS], dtype=torch.float32, device=D)

    def make_tensors(np):
        return weights_list + [
            hidden.clone(),
            torch.zeros(Q_DIM, dtype=torch.bfloat16, device=D),
            torch.zeros(HIDDEN_DIM, dtype=torch.bfloat16, device=D),
            torch.zeros(INTERMEDIATE_DIM, dtype=torch.bfloat16, device=D),
            torch.zeros(VOCAB_SIZE, dtype=torch.bfloat16, device=D),
            k_cache.clone(), v_cache.clone(), rope_cos, rope_sin,
            pos_id_t, scale_t, eps_t,
            torch.zeros(NUM_ATTENTION_HEADS, np, HEAD_DIM, dtype=torch.float32, device=D),
            torch.zeros(NUM_ATTENTION_HEADS, ((np + 15) // 16) * 16, dtype=torch.float32, device=D),
        ]

    def run(np, num_layers=NUM_LAYERS):
        sched = schedule_decode(sm_count=sm_count, num_partitions=np, num_layers=num_layers)
        d = Dispatcher(*sched, use_jit_cache=False)
        t = make_tensors(np)
        d(*t)
        return t[13].clone()  # logits

    print(f"\nReduction correctness (1 vs {num_partitions} partitions, seq_len={pos_id+1}):")

    for nl in [1, 16]:
        logits_1 = run(1, num_layers=nl)
        logits_n = run(num_partitions, num_layers=nl)
        diff = (logits_1.float() - logits_n.float()).abs()
        print(f"\n  {nl}-layer: max_diff={diff.max().item():.4f}  mean_diff={diff.mean().item():.6f}  argmax_match={logits_1.argmax().item() == logits_n.argmax().item()}")


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


def benchmark_tok_per_sec(prompt="Hello, my name is", max_new_tokens=200, num_samples=5, warmup=5, chunk_size=None):
    """tok/s with HF weights + greedy decode.

    chunk_size: if set, enables attention reduction. At each decode step,
    num_partitions = ceil(seq_len / chunk_size). When num_partitions > 1,
    attention is split across multiple SMs with a reduction step.
    """
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from megakittens.dispatcher import _pack_instructions_per_sm

    sm_count = get_sm_count()
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

    # Compute max num_partitions across all decode steps
    max_seq_len_needed = prompt_len + max_new_tokens
    if chunk_size is not None:
        max_np = max(1, math.ceil(max_seq_len_needed / chunk_size))
    else:
        max_np = 1

    k_cache = torch.zeros(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    v_cache = torch.zeros(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D)
    print("Prefilling KV cache...")
    with torch.inference_mode():
        last_hidden = _prefill_kv_cache(input_ids, weights, k_cache, v_cache, rope_cos, rope_sin)
    prefill_logits = weights["lm_head_weight"] @ _rmsnorm(last_hidden, weights["lm_head_norm_weight"], RMS_NORM_EPS)
    first_token = torch.argmax(prefill_logits)
    k_cache_snapshot = k_cache.clone()
    v_cache_snapshot = v_cache.clone()

    # Compile one kernel — includes all attention icodes when using reduction
    schedule = schedule_decode(sm_count=sm_count, num_partitions=1,
                               max_partitions=max_np if max_np > 1 else None)
    instruction_metas, tensor_metas, instructions, num_barriers, input_indices, output_indices = schedule

    dispatcher = Dispatcher(
        instruction_metas, tensor_metas, instructions, num_barriers,
        input_indices, output_indices,
        use_jit_cache=False,
    )

    # Pre-generate instruction tensors for each unique num_partitions
    instruction_tensors = None
    if chunk_size is not None and max_np > 1:
        instruction_tensors = {}
        for np_val in range(1, max_np + 1):
            _, _, insts, _, _, _ = schedule_decode(
                sm_count=sm_count, num_partitions=np_val, max_partitions=max_np)
            instruction_tensors[np_val] = _pack_instructions_per_sm(insts, sm_count, device=D)
        label = f"chunk_size={chunk_size}, max_partitions={max_np}"
    else:
        label = "no reduction"
    print(f"Attention mode: {label}")

    hidden_states = embed_weight[first_token].clone()
    logits = torch.zeros(VOCAB_SIZE, dtype=torch.bfloat16, device=D)
    pos_id_tensor = torch.tensor([prompt_len], dtype=torch.int32, device=D)
    attn_scale_tensor = torch.tensor([ATTN_SCALE], dtype=torch.float32, device=D)
    rms_norm_eps_tensor = torch.tensor([RMS_NORM_EPS], dtype=torch.float32, device=D)

    # Pre-allocate CPU-side buffer and cache GPU address for fast pos_id updates
    _pos_id_buf = (ctypes.c_int * 1)(0)
    _pos_id_gpu_ptr = pos_id_tensor.data_ptr()

    # same order as examples.llama1b.scheduler.T
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
        hidden_states,
        torch.zeros(Q_DIM, dtype=torch.bfloat16, device=D),
        torch.zeros(HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.zeros(INTERMEDIATE_DIM, dtype=torch.bfloat16, device=D),
        logits,
        k_cache,
        v_cache,
        rope_cos,
        rope_sin,
        pos_id_tensor,
        attn_scale_tensor,
        rms_norm_eps_tensor,
        torch.zeros(NUM_ATTENTION_HEADS, max_np, HEAD_DIM, dtype=torch.float32, device=D),
        torch.zeros(NUM_ATTENTION_HEADS, ((max_np + 15) // 16) * 16, dtype=torch.float32, device=D),
    ]

    model_size = sum(t.nelement() * t.element_size() for t in mk_tensors[:9])
    params = sum(t.nelement() for t in mk_tensors[:9])

    embedding = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_DIM, device=D, dtype=torch.bfloat16)
    embedding.weight.data.copy_(embed_weight)
    num_decode_tokens = max_new_tokens - 1
    output_tokens = torch.zeros(max_new_tokens, dtype=torch.long, device=D)

    def _decode_step(pos_id, input_token):
        hidden_states.copy_(embedding(input_token))
        _pos_id_buf[0] = pos_id
        stream = torch.cuda.current_stream().cuda_stream
        cuda_driver.cuMemcpyHtoDAsync(_pos_id_gpu_ptr, _pos_id_buf, 4, stream)
        if instruction_tensors is not None:
            seq_len = pos_id + 1
            np_val = max(1, math.ceil(seq_len / chunk_size))
            dispatcher.all_tensors[0] = instruction_tensors[np_val]
        dispatcher(*mk_tensors)
        return torch.argmax(logits, dim=-1)

    dispatcher(*mk_tensors)
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
    print(f"Attention mode: {label}")
    print(f"Average tokens/sec (decode only): {torch.mean(torch.tensor(decode_tokens_per_sec_list)).item():.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    benchmark_tok_per_sec(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        chunk_size=args.chunk_size,
        num_samples=args.num_samples,
        warmup=args.warmup,
    )
