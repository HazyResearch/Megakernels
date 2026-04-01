"""
Test rms_lm_head instruction in isolation.
Fused RMSNorm + LM head projection.
"""

import torch

from megakittens.jit.cuda_utils import initialize_cuda_context
from megakittens.dispatcher import Dispatcher, ScalarField
from megakittens.schema.device import Device
from megakittens.schema.dtype import DType
from megakittens.schema.instruction import Instruction, InstructionMeta
from megakittens.schema.tensor import TensorMeta
from megakittens.itypes.noop import Noop
from megakittens.itypes.rms_lm_head import RmsLmHead

initialize_cuda_context()

HIDDEN_DIM = 2048
VOCAB_SIZE = 128256
BLOCK_SIZE = 16
RMS_NORM_EPS = 1e-5
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Tensor indices
T_HIDDEN = 0
T_NORM_W = 1
T_LM_HEAD_W = 2
T_LOGITS = 3

SCALARS = [
    ScalarField("pos_id", "unsigned int", 4, 4),
    ScalarField("attn_scale", "float", 4, 4),
    ScalarField("rms_norm_eps", "float", 4, 4),
]


def rmsnorm(x, weight, eps):
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    normed = x_float * torch.rsqrt(variance + eps)
    return (weight.float() * normed).to(x.dtype)


def test_rms_lm_head():
    torch.manual_seed(42)
    device = Device(type="cuda", index=0)

    x = torch.randn(HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    norm_weight = torch.randn(HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    lm_head_weight = torch.randn(VOCAB_SIZE, HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    logits = torch.zeros(VOCAB_SIZE, dtype=DTYPE, device=DEVICE)

    # PyTorch reference
    normed = rmsnorm(x, norm_weight, RMS_NORM_EPS)
    expected = lm_head_weight @ normed

    num_blocks = VOCAB_SIZE // BLOCK_SIZE  # 8016

    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=device),
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=device),
        TensorMeta(dtype=DType.bf16, shape=(VOCAB_SIZE, HIDDEN_DIM), device=device),
        TensorMeta(dtype=DType.bf16, shape=(VOCAB_SIZE,), device=device),
    ]

    noop_itype = Noop()
    lm_head_itype = RmsLmHead(n=HIDDEN_DIM)

    instruction_metas = [
        InstructionMeta(icode=0, itype=noop_itype, src_tensors=(), dst_tensors=()),
        InstructionMeta(icode=1, itype=lm_head_itype,
                        src_tensors=(T_HIDDEN, T_NORM_W, T_LM_HEAD_W),
                        dst_tensors=(T_LOGITS,)),
    ]

    noop = Instruction(
        icode=0, src_tensors=(), dst_tensors=(), indices=(),
        src_barriers=(), src_barrier_targets=(),
        num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
        dst_barriers=(),
    )

    sm_count = 32
    instructions = []
    for sm in range(sm_count):
        start = round(sm * num_blocks / sm_count)
        end = round((sm + 1) * num_blocks / sm_count)
        instructions.append(Instruction(
            icode=1,
            src_tensors=(T_HIDDEN, T_NORM_W, T_LM_HEAD_W),
            dst_tensors=(T_LOGITS,),
            indices=(start, end),
            src_barriers=(),
            src_barrier_targets=(),
            num_input_barriers=0,
            num_reuse_barriers=0,
            num_dst_barriers=0,
            dst_barriers=(),
        ))

    while len(instructions) % 2 != 0:
        instructions.append(noop)

    input_indices = (T_HIDDEN, T_NORM_W, T_LM_HEAD_W, T_LOGITS)
    output_indices = (T_LOGITS,)

    dispatcher = Dispatcher(
        instruction_metas, tensor_metas, instructions,
        num_barriers=0,
        input_tensor_indices=input_indices,
        output_tensor_indices=output_indices,
        use_jit_cache=False,
        scalar_fields=SCALARS,
    )

    result = dispatcher(x, norm_weight, lm_head_weight, logits, 0, 0.125, RMS_NORM_EPS)
    torch.cuda.synchronize()

    diff = (result.float() - expected.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    assert mean_diff < 4.0, f"rms_lm_head failed: mean_diff={mean_diff}"
    print("PASS")


if __name__ == "__main__":
    test_rms_lm_head()
