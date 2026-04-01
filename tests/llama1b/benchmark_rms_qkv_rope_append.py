"""Benchmark rms_qkv_rope_append instruction."""

import torch

from megakittens.jit.cuda_utils import get_sm_count, initialize_cuda_context
from megakittens.dispatcher import Dispatcher, ScalarField
from megakittens.schema.device import Device
from megakittens.schema.dtype import DType
from megakittens.schema.instruction import Instruction, InstructionMeta
from megakittens.schema.tensor import TensorMeta
from megakittens.itypes.noop import Noop
from megakittens.itypes.rms_qkv_rope_append import RmsQkvRopeAppend

initialize_cuda_context()

HIDDEN_DIM = 2048
NUM_LAYERS = 16
NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64
QKV_DIM = (NUM_ATTENTION_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
Q_DIM = NUM_ATTENTION_HEADS * HEAD_DIM
MAX_SEQ_LEN = 512
BLOCK_SIZE = 16

SCALARS = [
    ScalarField("pos_id", "unsigned int", 4, 4),
    ScalarField("attn_scale", "float", 4, 4),
    ScalarField("rms_norm_eps", "float", 4, 4),
]


def build_dispatcher(sm_count):
    device = Device(type="cuda", index=0)

    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=device),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, HIDDEN_DIM), device=device),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, QKV_DIM, HIDDEN_DIM), device=device),
        TensorMeta(dtype=DType.fp32, shape=(MAX_SEQ_LEN, HEAD_DIM), device=device),
        TensorMeta(dtype=DType.fp32, shape=(MAX_SEQ_LEN, HEAD_DIM), device=device),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), device=device),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), device=device),
        TensorMeta(dtype=DType.bf16, shape=(Q_DIM,), device=device),
    ]

    noop_itype = Noop()
    qkv_itype = RmsQkvRopeAppend(n=HIDDEN_DIM)

    instruction_metas = [
        InstructionMeta(icode=0, itype=noop_itype, src_tensors=(), dst_tensors=()),
        InstructionMeta(icode=1, itype=qkv_itype,
                        src_tensors=(0, 1, 2, 3, 4, 5, 6),
                        dst_tensors=(7,)),
    ]

    num_blocks = QKV_DIM // BLOCK_SIZE
    noop = Instruction(
        icode=0, src_tensors=(), dst_tensors=(), indices=(),
        src_barriers=(), src_barrier_targets=(),
        num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
        dst_barriers=(),
    )

    instructions = []
    for sm in range(sm_count):
        start = round(sm * num_blocks / sm_count)
        end = round((sm + 1) * num_blocks / sm_count)
        instructions.append(Instruction(
            icode=1,
            src_tensors=(0, 1, 2, 3, 4, 5, 6),
            dst_tensors=(7,),
            indices=(0, start, end),
            src_barriers=(), src_barrier_targets=(),
            num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
            dst_barriers=(),
        ))
    while len(instructions) % 2 != 0:
        instructions.append(noop)

    return Dispatcher(
        instruction_metas, tensor_metas, instructions,
        num_barriers=0,
        input_tensor_indices=tuple(range(8)),
        output_tensor_indices=(7, 5, 6),
        use_jit_cache=False,
        scalar_fields=SCALARS,
    )


def allocate_tensors():
    D = "cuda"
    return [
        torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(NUM_LAYERS, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(NUM_LAYERS, QKV_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=D),
        torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=D),
        torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=D),
        torch.zeros(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D),
        torch.zeros(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=D),
        torch.zeros(Q_DIM, dtype=torch.bfloat16, device=D),
    ]


def benchmark_rms_qkv_rope_append(warmup=500, iters=1000):
    sm_count = get_sm_count()
    dispatcher = build_dispatcher(sm_count)
    tensors = allocate_tensors()

    for _ in range(warmup):
        dispatcher(*tensors, 0, 0.125, 1e-5)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dispatcher(*tensors, 0, 0.125, 1e-5)
    end.record()
    torch.cuda.synchronize()

    us = start.elapsed_time(end) / iters * 1000
    print(f"rms_qkv_rope_append: {us:.1f} us  ({sm_count} SMs, {iters} iters)")


if __name__ == "__main__":
    benchmark_rms_qkv_rope_append()
