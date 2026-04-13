"""
Test attention_partial instruction in isolation.
Single-partition decode attention: softmax(Q @ K^T * scale) @ V per KV head.
"""

import math

import torch
import torch.nn.functional as F

from megakittens.jit.cuda_utils import initialize_cuda_context
from megakittens.dispatcher import Dispatcher, ScalarField
from megakittens.schema.device import Device
from megakittens.schema.dtype import DType
from megakittens.schema.instruction import Instruction, InstructionMeta
from megakittens.schema.tensor import TensorMeta
from megakittens.itypes.llama1b.attention_partial import AttentionPartial
from megakittens.itypes.noop import Noop

initialize_cuda_context()

HIDDEN_DIM = 2048
NUM_LAYERS = 16
NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64
GQA_RATIO = NUM_ATTENTION_HEADS // NUM_KV_HEADS  # 4
DEVICE = "cuda"
DTYPE = torch.bfloat16


# Tensor indices
T_Q = 0
T_K_CACHE = 1
T_V_CACHE = 2
T_ATTN_OUT = 3

SCALARS = [
    ScalarField("pos_id", "unsigned int", 4, 4),
    ScalarField("attn_scale", "float", 4, 4),
    ScalarField("rms_norm_eps", "float", 4, 4),
]


def test_attention_partial():
    torch.manual_seed(42)
    device = Device(type="cuda", index=0)

    seq_len = 37
    pos_id = seq_len - 1
    max_seq_len = 128
    attn_scale = 1.0 / math.sqrt(HEAD_DIM)
    layer_idx = 2

    # Random test data
    q = torch.randn(NUM_ATTENTION_HEADS * HEAD_DIM, dtype=DTYPE, device=DEVICE)
    k_cache = torch.randn(NUM_LAYERS, max_seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v_cache = torch.randn(NUM_LAYERS, max_seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    attn_out = torch.zeros(NUM_ATTENTION_HEADS * HEAD_DIM, dtype=DTYPE, device=DEVICE)

    # PyTorch reference
    q_heads = q.view(NUM_ATTENTION_HEADS, HEAD_DIM)
    expected = torch.zeros_like(attn_out)
    for kv_head in range(NUM_KV_HEADS):
        k_cached = k_cache[layer_idx, :seq_len, kv_head]  # [seq_len, 64]
        v_cached = v_cache[layer_idx, :seq_len, kv_head]  # [seq_len, 64]
        gqa_start = kv_head * GQA_RATIO
        gqa_end = gqa_start + GQA_RATIO
        for q_head in range(gqa_start, gqa_end):
            scores = (q_heads[q_head] @ k_cached.T) * attn_scale
            w = F.softmax(scores.float(), dim=-1).to(DTYPE)
            out = w @ v_cached
            expected[q_head * HEAD_DIM:(q_head + 1) * HEAD_DIM] = out

    # Build schedule: one instruction per KV head
    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(NUM_ATTENTION_HEADS * HEAD_DIM,), device=device),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, max_seq_len, NUM_KV_HEADS, HEAD_DIM), device=device),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, max_seq_len, NUM_KV_HEADS, HEAD_DIM), device=device),
        TensorMeta(dtype=DType.bf16, shape=(NUM_ATTENTION_HEADS * HEAD_DIM,), device=device),
    ]

    noop_itype = Noop()
    attn_itype = AttentionPartial()

    instruction_metas = [
        InstructionMeta(icode=0, itype=noop_itype, src_tensors=(), dst_tensors=()),
        InstructionMeta(icode=1, itype=attn_itype,
                        src_tensors=(T_Q, T_K_CACHE, T_V_CACHE),
                        dst_tensors=(T_ATTN_OUT,)),
    ]

    instructions = []
    for kv_head in range(NUM_KV_HEADS):
        instructions.append(Instruction(
            icode=1,
            src_tensors=(T_Q, T_K_CACHE, T_V_CACHE),
            dst_tensors=(T_ATTN_OUT,),
            indices=(layer_idx, kv_head),
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

    input_indices = (T_Q, T_K_CACHE, T_V_CACHE, T_ATTN_OUT)
    output_indices = (T_ATTN_OUT,)

    dispatcher = Dispatcher(
        instruction_metas, tensor_metas, instructions,
        num_barriers=0,
        input_tensor_indices=input_indices,
        output_tensor_indices=output_indices,
        use_jit_cache=False,
        scalar_fields=SCALARS,
    )

    result = dispatcher(q, k_cache, v_cache, attn_out, pos_id, attn_scale, 1e-5)
    torch.cuda.synchronize()

    diff = (result.float() - expected.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    assert torch.allclose(result.float(), expected.float(), atol=1e-1, rtol=1e-1), \
        f"attention_partial failed: max_diff={max_diff}, mean_diff={mean_diff}"
    print("PASS")


if __name__ == "__main__":
    test_attention_partial()
