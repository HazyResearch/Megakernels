"""
Test rms_qkv_rope_append instruction in isolation.
RMSNorm + QKV matvec + RoPE + KV cache write.
"""

import torch

from megakittens.jit.cuda_utils import initialize_cuda_context
from megakittens.dispatcher import Dispatcher, ScalarField
from megakittens.schema.device import Device
from megakittens.schema.dtype import DType
from megakittens.schema.instruction import Instruction, InstructionMeta
from megakittens.schema.tensor import TensorMeta
from megakittens.itypes.noop import Noop
from megakittens.itypes.llama1b.rms_qkv_rope_append import RmsQkvRopeAppend

initialize_cuda_context()

HIDDEN_DIM = 2048
NUM_LAYERS = 16
NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64
QKV_DIM = (NUM_ATTENTION_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM  # 3072
Q_DIM = NUM_ATTENTION_HEADS * HEAD_DIM  # 2048
K_DIM = NUM_KV_HEADS * HEAD_DIM  # 512
BLOCK_SIZE = 16
DEVICE = "cuda"
DTYPE = torch.bfloat16
RMS_NORM_EPS = 1e-5

# Tensor indices
T_HIDDEN = 0
T_NORM_W = 1
T_QKV_W = 2
T_COS = 3
T_SIN = 4
T_K_CACHE = 5
T_V_CACHE = 6
T_Q_OUT = 7

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


def test_rms_qkv_rope_append():
    torch.manual_seed(42)
    device = Device(type="cuda", index=0)

    layer_idx = 2
    pos_id = 37
    max_seq_len = 128

    # Random test data
    hidden = torch.randn(HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    norm_w = torch.randn(NUM_LAYERS, HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    qkv_w = torch.randn(NUM_LAYERS, QKV_DIM, HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    rope_cos = torch.randn(max_seq_len, HEAD_DIM, dtype=torch.float32, device=DEVICE)
    rope_sin = torch.randn(max_seq_len, HEAD_DIM, dtype=torch.float32, device=DEVICE)
    k_cache = torch.zeros(NUM_LAYERS, max_seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v_cache = torch.zeros(NUM_LAYERS, max_seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    q_out = torch.zeros(Q_DIM, dtype=DTYPE, device=DEVICE)

    # --- PyTorch reference ---
    normed = rmsnorm(hidden, norm_w[layer_idx], RMS_NORM_EPS)
    qkv = qkv_w[layer_idx] @ normed  # [3072]
    q_ref = qkv[:Q_DIM].view(NUM_ATTENTION_HEADS, HEAD_DIM)
    k_ref = qkv[Q_DIM:Q_DIM + K_DIM].view(NUM_KV_HEADS, HEAD_DIM)
    v_ref = qkv[Q_DIM + K_DIM:].view(NUM_KV_HEADS, HEAD_DIM)

    cos_pos = rope_cos[pos_id]  # [64]
    sin_pos = rope_sin[pos_id]  # [64]
    q_ref = apply_rope_interleaved(q_ref, cos_pos, sin_pos).flatten()
    k_ref = apply_rope_interleaved(k_ref, cos_pos, sin_pos)
    # v_ref unchanged

    # --- Build schedule ---
    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=device),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, HIDDEN_DIM), device=device),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, QKV_DIM, HIDDEN_DIM), device=device),
        TensorMeta(dtype=DType.fp32, shape=(max_seq_len, HEAD_DIM), device=device),
        TensorMeta(dtype=DType.fp32, shape=(max_seq_len, HEAD_DIM), device=device),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, max_seq_len, NUM_KV_HEADS, HEAD_DIM), device=device),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, max_seq_len, NUM_KV_HEADS, HEAD_DIM), device=device),
        TensorMeta(dtype=DType.bf16, shape=(Q_DIM,), device=device),
    ]

    noop_itype = Noop()
    qkv_itype = RmsQkvRopeAppend(n=HIDDEN_DIM)

    instruction_metas = [
        InstructionMeta(icode=0, itype=noop_itype, src_tensors=(), dst_tensors=()),
        InstructionMeta(icode=1, itype=qkv_itype,
                        src_tensors=(T_HIDDEN, T_NORM_W, T_QKV_W,
                                     T_COS, T_SIN, T_K_CACHE, T_V_CACHE),
                        dst_tensors=(T_Q_OUT,)),
    ]

    # Distribute blocks across instructions (simulate 4 SMs)
    num_blocks = QKV_DIM // BLOCK_SIZE  # 192
    sm_count = 4
    instructions = []
    for sm in range(sm_count):
        start = round(sm * num_blocks / sm_count)
        end = round((sm + 1) * num_blocks / sm_count)
        instructions.append(Instruction(
            icode=1,
            src_tensors=(T_HIDDEN, T_NORM_W, T_QKV_W,
                         T_COS, T_SIN, T_K_CACHE, T_V_CACHE),
            dst_tensors=(T_Q_OUT,),
            indices=(layer_idx, start, end),
            src_barriers=(),
            src_barrier_targets=(),
            num_input_barriers=0,
            num_reuse_barriers=0,
            num_dst_barriers=0,
            dst_barriers=(),
        ))

    # Pad to cluster size
    noop = Instruction(
        icode=0, src_tensors=(), dst_tensors=(), indices=(),
        src_barriers=(), src_barrier_targets=(),
        num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
        dst_barriers=(),
    )
    while len(instructions) % 2 != 0:
        instructions.append(noop)

    input_indices = tuple(range(8))
    output_indices = (T_Q_OUT, T_K_CACHE, T_V_CACHE)

    dispatcher = Dispatcher(
        instruction_metas, tensor_metas, instructions,
        num_barriers=0,
        input_tensor_indices=input_indices,
        output_tensor_indices=output_indices,
        use_jit_cache=False,
        scalar_fields=SCALARS,
    )

    dispatcher(hidden, norm_w, qkv_w, rope_cos, rope_sin,
               k_cache, v_cache, q_out,
               pos_id, 0.125, RMS_NORM_EPS)
    torch.cuda.synchronize()

    # --- Check Q ---
    q_diff = (q_out.float() - q_ref.float()).abs()
    q_max = q_diff.max().item()
    q_mean = q_diff.mean().item()
    print(f"Q: max_diff={q_max:.4f}, mean_diff={q_mean:.6f}")

    # --- Check K cache ---
    k_actual = k_cache[layer_idx, pos_id]  # [8, 64]
    k_diff = (k_actual.float() - k_ref.float()).abs()
    k_max = k_diff.max().item()
    k_mean = k_diff.mean().item()
    print(f"K: max_diff={k_max:.4f}, mean_diff={k_mean:.6f}")

    # --- Check V cache ---
    v_actual = v_cache[layer_idx, pos_id]  # [8, 64]
    v_diff = (v_actual.float() - v_ref.float()).abs()
    v_max = v_diff.max().item()
    v_mean = v_diff.mean().item()
    print(f"V: max_diff={v_max:.4f}, mean_diff={v_mean:.6f}")

    assert q_max < 4.0, f"Q failed: max_diff={q_max}"
    assert k_max < 4.0, f"K failed: max_diff={k_max}"
    assert v_max < 4.0, f"V failed: max_diff={v_max}"

    print("PASS")


if __name__ == "__main__":
    test_rms_qkv_rope_append()
