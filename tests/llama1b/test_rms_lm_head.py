"""
Standalone test for the RmsLmHead kernel.
Schedules ONLY lm_head instructions (no other layers), feeds known inputs,
compares against PyTorch reference.
"""

import torch

from megakittens.jit.cuda_utils import get_sm_count, initialize_cuda_context
from megakittens.dispatcher import Dispatcher, ScalarField
from megakittens.itypes.llama1b.rms_lm_head import RmsLmHead
from megakittens.itypes.noop import Noop
from megakittens.schema.device import Device
from megakittens.schema.dtype import DType
from megakittens.schema.instruction import Instruction, InstructionMeta
from megakittens.schema.tensor import TensorMeta

initialize_cuda_context()

DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN_DIM = 2048
VOCAB_SIZE = 128256  # must be divisible by 16
BLOCK_SIZE = 16
RMS_NORM_EPS = 1e-5
CLUSTER_SIZE = 2

SCALARS = [
    ScalarField("pos_id", "unsigned int", 4, 4),
    ScalarField("attn_scale", "float", 4, 4),
    ScalarField("rms_norm_eps", "float", 4, 4),
]


class T:
    HIDDEN_STATES = 0
    NORM_WEIGHT = 1
    LM_HEAD_WEIGHT = 2
    LOGITS = 3
    COUNT = 4


def schedule_lm_head_only(sm_count):
    device = Device(type="cuda", index=0)
    bf16 = DType.bf16

    tensor_metas = [
        TensorMeta(dtype=bf16, shape=(HIDDEN_DIM,), device=device),
        TensorMeta(dtype=bf16, shape=(HIDDEN_DIM,), device=device),
        TensorMeta(dtype=bf16, shape=(VOCAB_SIZE, HIDDEN_DIM), device=device),
        TensorMeta(dtype=bf16, shape=(VOCAB_SIZE,), device=device),
    ]

    noop_meta = InstructionMeta(icode=0, itype=Noop(), src_tensors=(), dst_tensors=())
    lm_head_meta = InstructionMeta(
        icode=1,
        itype=RmsLmHead(n=HIDDEN_DIM),
        src_tensors=(T.HIDDEN_STATES, T.NORM_WEIGHT, T.LM_HEAD_WEIGHT),
        dst_tensors=(T.LOGITS,),
    )
    instruction_metas = [noop_meta, lm_head_meta]

    num_blocks = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    instructions = []
    for sm in range(sm_count):
        start = round(sm * num_blocks / sm_count)
        end = round((sm + 1) * num_blocks / sm_count)
        instructions.append(Instruction(
            icode=1,
            src_tensors=(T.HIDDEN_STATES, T.NORM_WEIGHT, T.LM_HEAD_WEIGHT),
            dst_tensors=(T.LOGITS,),
            indices=(start, end),
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
    while len(instructions) % CLUSTER_SIZE != 0:
        instructions.append(noop)

    return (
        instruction_metas,
        tensor_metas,
        instructions,
        0,  # num_barriers
        tuple(range(T.COUNT)),
        (T.LOGITS,),
    )


def rmsnorm(x, weight, eps):
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    return (x_f * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


@torch.inference_mode()
def test_rms_lm_head():
    torch.manual_seed(42)

    hidden_states = torch.randn(HIDDEN_DIM, device=DEVICE, dtype=DTYPE)
    norm_weight = torch.randn(HIDDEN_DIM, device=DEVICE, dtype=DTYPE)
    lm_head_weight = torch.randn(VOCAB_SIZE, HIDDEN_DIM, device=DEVICE, dtype=DTYPE)
    logits = torch.zeros(VOCAB_SIZE, device=DEVICE, dtype=DTYPE)

    # Reference
    normed = rmsnorm(hidden_states, norm_weight, RMS_NORM_EPS)
    ref_logits = (lm_head_weight @ normed).float()

    # Megakernel
    sm_count = get_sm_count()
    schedule = schedule_lm_head_only(sm_count)
    instruction_metas, tensor_metas, instructions, num_barriers, input_indices, output_indices = schedule

    dispatcher = Dispatcher(
        instruction_metas, tensor_metas, instructions, num_barriers,
        input_indices, output_indices,
        use_jit_cache=False,
        scalar_fields=SCALARS,
    )

    tensors = [hidden_states, norm_weight, lm_head_weight, logits]
    mk_logits = dispatcher(*tensors, 0, 0.0, RMS_NORM_EPS)
    torch.cuda.synchronize()

    mk_logits_f = mk_logits.float()
    diff = (mk_logits_f - ref_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    mk_top = mk_logits_f.topk(10).indices.tolist()
    ref_top = ref_logits.topk(10).indices.tolist()

    print(f"max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f}")
    print(f"MK  top-10: {mk_top}")
    print(f"Ref top-10: {ref_top}")
    print(f"Top match: {mk_top[0] == ref_top[0]}")

    assert mk_top[0] == ref_top[0], f"Top token mismatch: MK={mk_top[0]} vs ref={ref_top[0]}"
    print("PASS")


if __name__ == "__main__":
    test_rms_lm_head()
