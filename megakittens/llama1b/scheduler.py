"""
Llama-1B decode scheduler.

Produces a flat instruction list with barrier dependencies,
matching the megakittens Instruction/InstructionMeta format.
"""

from __future__ import annotations

from ..itypes.attention_partial import AttentionPartial
from ..itypes.noop import Noop
from ..itypes.matvec_adds import MatVecAdds
from ..itypes.rms_lm_head import RmsLmHead
from ..itypes.rms_qkv_rope_append import RmsQkvRopeAppend
from ..itypes.rms_upgate_silu import RmsUpgateSilu
from ..schema.device import Device
from ..schema.dtype import DType
from ..schema.instruction import Instruction, InstructionMeta
from ..schema.tensor import TensorMeta


# -- Llama-3.2-1B constants --------------------------------------------------

NUM_LAYERS = 16
HIDDEN_DIM = 2048
INTERMEDIATE_DIM = 8192
NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64
VOCAB_SIZE = 128256
RMS_NORM_EPS = 1e-5
MATVEC_BLOCK_SIZE = 16
MAX_SEQ_LEN = 512

# QKV layout: [Q rows | K rows | V rows] = [2048 | 512 | 512] = 3072
Q_DIM = NUM_ATTENTION_HEADS * HEAD_DIM      # 2048
K_DIM = NUM_KV_HEADS * HEAD_DIM             # 512
V_DIM = NUM_KV_HEADS * HEAD_DIM             # 512
QKV_DIM = Q_DIM + K_DIM + V_DIM             # 3072


# -- Tensor index assignments ------------------------------------------------
# Every GPU buffer gets a fixed index. Instructions reference these indices
# in their src_tensors / dst_tensors fields, and the dispatcher generates
# MKGlobals with tensor_0, tensor_1, ... in this order.

class T:
    """Tensor index constants."""
    QKV_WEIGHTS = 0          # [NUM_LAYERS, QKV_DIM, HIDDEN_DIM]
    O_WEIGHTS = 1            # [NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM]
    ATTN_NORM_WEIGHTS = 2    # [NUM_LAYERS, HIDDEN_DIM]
    MLP_NORM_WEIGHTS = 3     # [NUM_LAYERS, HIDDEN_DIM]
    UP_WEIGHTS = 4           # [NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM]
    GATE_WEIGHTS = 5         # [NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM]
    DOWN_WEIGHTS = 6         # [NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM]
    LM_HEAD_NORM_WEIGHT = 7  # [HIDDEN_DIM]
    LM_HEAD_WEIGHT = 8       # [VOCAB_SIZE, HIDDEN_DIM]
    HIDDEN_STATES = 9        # [HIDDEN_DIM]
    Q_POST_ROPE = 10         # [Q_DIM]
    ATTN_OUT = 11            # [HIDDEN_DIM]
    SILU_OUT = 12            # [INTERMEDIATE_DIM]
    LOGITS = 13              # [VOCAB_SIZE]
    K_CACHE = 14             # [NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM]
    V_CACHE = 15             # [NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM]
    ROPE_COS = 16            # [MAX_SEQ_LEN, HEAD_DIM]
    ROPE_SIN = 17            # [MAX_SEQ_LEN, HEAD_DIM]
    COUNT = 18


# -- Schedule result type -----------------------------------------------------

# The scheduler returns a 6-tuple matching the existing Dispatcher constructor:
# (instruction_metas, tensor_metas, instructions, num_barriers,
#  input_tensor_indices, output_tensor_indices)
ScheduleResult = tuple[
    list[InstructionMeta],
    list[TensorMeta],
    list[Instruction],
    int,
    tuple[int, ...],
    tuple[int, ...],
]


# -- Helpers ------------------------------------------------------------------

CLUSTER_SIZE = 2
GQA_RATIO = NUM_ATTENTION_HEADS // NUM_KV_HEADS  # 4
BLOCKS_PER_HEAD = HEAD_DIM // MATVEC_BLOCK_SIZE   # 4

# -- Fine-grained barrier layout -----------------------------------------------
# 3D logical structure: barriers[layer][opcode_slot][sub_idx]
# Flattened to a 1D int32 array.
NUM_BARRIER_SLOTS = 6   # qkv, attn, attn_red, oproj, upgate, downproj
MAX_SUB_BARRIERS = 48   # largest: QKV with 32 Q + 8 K + 8 V = 48

# Sub-barrier counts per opcode slot
QKV_SUB_BARRIERS = NUM_ATTENTION_HEADS + 2 * NUM_KV_HEADS  # 48
ATTN_SUB_BARRIERS = NUM_ATTENTION_HEADS                     # 32
ATTN_RED_SUB_BARRIERS = 1
OPROJ_SUB_BARRIERS = 1
UPGATE_SUB_BARRIERS = INTERMEDIATE_DIM // HIDDEN_DIM        # 4
DOWNPROJ_SUB_BARRIERS = 1

# Target counts for fine-grained barriers
QKV_TARGET_PER_SUB = BLOCKS_PER_HEAD                                    # 4
UPGATE_TARGET_PER_SUB = HIDDEN_DIM // MATVEC_BLOCK_SIZE                 # 128
ATTN_RED_TARGET = NUM_ATTENTION_HEADS                                   # 32
OPROJ_TARGET = HIDDEN_DIM // MATVEC_BLOCK_SIZE                          # 128
DOWNPROJ_INSTS_PER_CHUNK = 32  # 128 blocks / 32 = 4 blocks each
DOWNPROJ_TARGET = INTERMEDIATE_DIM // MATVEC_BLOCK_SIZE  # 512


def _barrier_index(layer_idx: int, opcode_slot: int, sub_idx: int = 0) -> int:
    return layer_idx * NUM_BARRIER_SLOTS * MAX_SUB_BARRIERS + opcode_slot * MAX_SUB_BARRIERS + sub_idx


_noop_inst_meta = InstructionMeta(icode=0, itype=Noop(), src_tensors=(), dst_tensors=())

_noop = Instruction(
    icode=0, src_tensors=(), dst_tensors=(), indices=(),
    src_barriers=(), src_barrier_targets=(),
    num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
    dst_barriers=(),
)


def _pad_to_cluster(instructions: list[Instruction]) -> None:
    while len(instructions) % CLUSTER_SIZE != 0:
        instructions.append(_noop)


# -- Tensor metadata ----------------------------------------------------------

def _make_tensor_metas(device: Device) -> list[TensorMeta]:
    bf16 = DType.bf16
    fp32 = DType.fp32
    metas = [None] * T.COUNT

    metas[T.QKV_WEIGHTS] = TensorMeta(dtype=bf16, shape=(NUM_LAYERS, QKV_DIM, HIDDEN_DIM), device=device)
    metas[T.O_WEIGHTS] = TensorMeta(dtype=bf16, shape=(NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM), device=device)
    metas[T.ATTN_NORM_WEIGHTS] = TensorMeta(dtype=bf16, shape=(NUM_LAYERS, HIDDEN_DIM), device=device)
    metas[T.MLP_NORM_WEIGHTS] = TensorMeta(dtype=bf16, shape=(NUM_LAYERS, HIDDEN_DIM), device=device)
    metas[T.UP_WEIGHTS] = TensorMeta(dtype=bf16, shape=(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM), device=device)
    metas[T.GATE_WEIGHTS] = TensorMeta(dtype=bf16, shape=(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM), device=device)
    metas[T.DOWN_WEIGHTS] = TensorMeta(dtype=bf16, shape=(NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM), device=device)
    metas[T.LM_HEAD_NORM_WEIGHT] = TensorMeta(dtype=bf16, shape=(HIDDEN_DIM,), device=device)
    metas[T.LM_HEAD_WEIGHT] = TensorMeta(dtype=bf16, shape=(VOCAB_SIZE, HIDDEN_DIM), device=device)
    metas[T.HIDDEN_STATES] = TensorMeta(dtype=bf16, shape=(HIDDEN_DIM,), device=device)
    metas[T.Q_POST_ROPE] = TensorMeta(dtype=bf16, shape=(Q_DIM,), device=device)
    metas[T.ATTN_OUT] = TensorMeta(dtype=bf16, shape=(HIDDEN_DIM,), device=device)
    metas[T.SILU_OUT] = TensorMeta(dtype=bf16, shape=(INTERMEDIATE_DIM,), device=device)
    metas[T.LOGITS] = TensorMeta(dtype=bf16, shape=(VOCAB_SIZE,), device=device)
    metas[T.K_CACHE] = TensorMeta(dtype=bf16, shape=(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), device=device)
    metas[T.V_CACHE] = TensorMeta(dtype=bf16, shape=(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), device=device)
    metas[T.ROPE_COS] = TensorMeta(dtype=fp32, shape=(MAX_SEQ_LEN, HEAD_DIM), device=device)
    metas[T.ROPE_SIN] = TensorMeta(dtype=fp32, shape=(MAX_SEQ_LEN, HEAD_DIM), device=device)

    return metas


# -- Per-instruction-group schedulers -----------------------------------------

def _schedule_qkv(
    sm_count: int,
    layer_idx: int,
    icode: int,
    prev_barrier: int | None,
    prev_barrier_target: int,
) -> list[Instruction]:
    barrier_base = _barrier_index(layer_idx, 0, 0)
    num_blocks = QKV_DIM // MATVEC_BLOCK_SIZE  # 192
    instructions = []
    for sm in range(sm_count):
        start = round(sm * num_blocks / sm_count)
        end = round((sm + 1) * num_blocks / sm_count)
        src_barriers = (prev_barrier,) if prev_barrier is not None else ()
        src_targets = (prev_barrier_target,) if prev_barrier is not None else ()
        instructions.append(Instruction(
            icode=icode,
            src_tensors=(T.HIDDEN_STATES, T.ATTN_NORM_WEIGHTS, T.QKV_WEIGHTS,
                         T.ROPE_COS, T.ROPE_SIN, T.K_CACHE, T.V_CACHE),
            dst_tensors=(T.Q_POST_ROPE,),
            indices=(layer_idx, start, end, barrier_base),
            src_barriers=src_barriers,
            src_barrier_targets=src_targets,
            num_input_barriers=len(src_barriers),
            num_reuse_barriers=0,
            num_dst_barriers=0,
            dst_barriers=(),
        ))
    _pad_to_cluster(instructions)
    return instructions


def _schedule_attention(
    layer_idx: int,
    icode: int,
) -> list[Instruction]:
    # No attention_reduction step currently (1 partial per head),
    # so attention signals the attn_red barrier (slot 2) directly.
    # This matches the reference's skip_attn_reduction path.
    attn_red_barrier = _barrier_index(layer_idx, 2, 0)
    instructions = []
    for kv_head in range(NUM_KV_HEADS):
        # Fine-grained consumer: wait on 6 QKV sub-barriers for this KV head
        # (4 Q heads + 1 K head + 1 V head), each with target = BLOCKS_PER_HEAD
        q_head_start = kv_head * GQA_RATIO
        src_barriers = tuple(
            _barrier_index(layer_idx, 0, q_head_start + h) for h in range(GQA_RATIO)
        ) + (
            _barrier_index(layer_idx, 0, NUM_ATTENTION_HEADS + kv_head),
            _barrier_index(layer_idx, 0, NUM_ATTENTION_HEADS + NUM_KV_HEADS + kv_head),
        )
        src_targets = (QKV_TARGET_PER_SUB,) * (GQA_RATIO + 2)
        instructions.append(Instruction(
            icode=icode,
            src_tensors=(T.Q_POST_ROPE, T.K_CACHE, T.V_CACHE),
            dst_tensors=(T.ATTN_OUT,),
            indices=(layer_idx, kv_head, attn_red_barrier),
            src_barriers=src_barriers,
            src_barrier_targets=src_targets,
            num_input_barriers=len(src_barriers),
            num_reuse_barriers=0,
            num_dst_barriers=0,
            dst_barriers=(),
        ))
    _pad_to_cluster(instructions)
    return instructions


def _schedule_o_proj(
    sm_count: int,
    layer_idx: int,
    icode: int,
) -> list[Instruction]:
    attn_red_barrier = _barrier_index(layer_idx, 2, 0)
    oproj_barrier = _barrier_index(layer_idx, 3, 0)
    num_blocks = HIDDEN_DIM // MATVEC_BLOCK_SIZE  # 128
    instructions = []
    for sm in range(num_blocks):
        instructions.append(Instruction(
            icode=icode,
            src_tensors=(T.ATTN_OUT, T.O_WEIGHTS),
            dst_tensors=(T.HIDDEN_STATES,),
            indices=(layer_idx, sm, sm + 1, 0),
            src_barriers=(attn_red_barrier,),
            src_barrier_targets=(ATTN_RED_TARGET,),
            num_input_barriers=1,
            num_reuse_barriers=0,
            num_dst_barriers=1,
            dst_barriers=(oproj_barrier,),
        ))
    _pad_to_cluster(instructions)
    return instructions


def _schedule_upgate(
    sm_count: int,
    layer_idx: int,
    icode: int,
) -> list[Instruction]:
    oproj_barrier = _barrier_index(layer_idx, 3, 0)
    upgate_barrier_base = _barrier_index(layer_idx, 4, 0)
    num_blocks = INTERMEDIATE_DIM // MATVEC_BLOCK_SIZE  # 512
    instructions = []
    for sm in range(sm_count):
        start = round(sm * num_blocks / sm_count)
        end = round((sm + 1) * num_blocks / sm_count)
        instructions.append(Instruction(
            icode=icode,
            src_tensors=(T.HIDDEN_STATES, T.MLP_NORM_WEIGHTS, T.UP_WEIGHTS, T.GATE_WEIGHTS),
            dst_tensors=(T.SILU_OUT,),
            indices=(layer_idx, start, end, upgate_barrier_base),
            src_barriers=(oproj_barrier,),
            src_barrier_targets=(OPROJ_TARGET,),
            num_input_barriers=1,
            num_reuse_barriers=0,
            num_dst_barriers=0,
            dst_barriers=(),
        ))
    _pad_to_cluster(instructions)
    return instructions


def _schedule_downproj(
    sm_count: int,
    layer_idx: int,
    icode: int,
) -> list[Instruction]:
    downproj_barrier = _barrier_index(layer_idx, 5, 0)
    num_blocks = HIDDEN_DIM // MATVEC_BLOCK_SIZE  # 128
    num_chunks = INTERMEDIATE_DIM // HIDDEN_DIM   # 4 reduction chunks
    instructions = []
    for chunk in range(num_chunks):
        reduction_col_offset = chunk * HIDDEN_DIM
        # Fine-grained consumer: wait on the upgate sub-barrier for this chunk
        upgate_sub = _barrier_index(layer_idx, 4, chunk)
        for i in range(DOWNPROJ_INSTS_PER_CHUNK):
            start = round(i * num_blocks / DOWNPROJ_INSTS_PER_CHUNK)
            end = round((i + 1) * num_blocks / DOWNPROJ_INSTS_PER_CHUNK)
            instructions.append(Instruction(
                icode=icode,
                src_tensors=(T.SILU_OUT, T.DOWN_WEIGHTS),
                dst_tensors=(T.HIDDEN_STATES,),
                indices=(layer_idx, start, end, reduction_col_offset),
                src_barriers=(upgate_sub,),
                src_barrier_targets=(UPGATE_TARGET_PER_SUB,),
                num_input_barriers=1,
                num_reuse_barriers=0,
                num_dst_barriers=1,
                dst_barriers=(downproj_barrier,),
            ))
    _pad_to_cluster(instructions)
    return instructions


def _schedule_lm_head(
    sm_count: int,
    num_layers: int,
    icode: int,
) -> list[Instruction]:
    prev_barrier = _barrier_index(num_layers - 1, 5, 0)
    prev_barrier_target = DOWNPROJ_TARGET
    num_blocks = VOCAB_SIZE // MATVEC_BLOCK_SIZE
    # round up for non-divisible vocab
    if VOCAB_SIZE % MATVEC_BLOCK_SIZE != 0:
        num_blocks += 1
    instructions = []
    for sm in range(sm_count):
        start = round(sm * num_blocks / sm_count)
        end = round((sm + 1) * num_blocks / sm_count)
        instructions.append(Instruction(
            icode=icode,
            src_tensors=(T.HIDDEN_STATES, T.LM_HEAD_NORM_WEIGHT, T.LM_HEAD_WEIGHT),
            dst_tensors=(T.LOGITS,),
            indices=(start, end),
            src_barriers=(prev_barrier,),
            src_barrier_targets=(prev_barrier_target,),
            num_input_barriers=1,
            num_reuse_barriers=0,
            num_dst_barriers=0,
            dst_barriers=(),
        ))
    _pad_to_cluster(instructions)
    return instructions


# -- Main entry point ---------------------------------------------------------

# Icode assignments (0 is reserved for noop)
ICODE_QKV = 1
ICODE_ATTENTION = 2
ICODE_O_PROJ = 3
ICODE_UPGATE = 4
ICODE_DOWNPROJ = 5
ICODE_LM_HEAD = 6


def schedule_decode(
    sm_count: int,
    num_layers: int = NUM_LAYERS,
    device: Device | None = None,
    noop: bool = False,
) -> ScheduleResult:
    """Build the full decode instruction schedule.

    Returns a 6-tuple that can be passed directly to the Dispatcher constructor.

    Args:
        sm_count: Number of SMs on the target GPU.
        num_layers: Number of transformer layers (default: 16).
        device: Target device (default: cuda:0).
        noop: If True, all instructions use icode=0 (noop) for plumbing tests.
    """
    if device is None:
        device = Device(type="cuda", index=0)

    tensor_metas = _make_tensor_metas(device)

    # Icode mapping — noop mode sets everything to 0
    icode_qkv = 0 if noop else ICODE_QKV
    icode_attn = 0 if noop else ICODE_ATTENTION
    icode_oproj = 0 if noop else ICODE_O_PROJ
    icode_upgate = 0 if noop else ICODE_UPGATE
    icode_downproj = 0 if noop else ICODE_DOWNPROJ
    icode_lmhead = 0 if noop else ICODE_LM_HEAD

    # Instruction metas (drive JIT codegen)
    _matvec_adds_itype = MatVecAdds(n=HIDDEN_DIM)
    _attention_partial_itype = AttentionPartial()
    _rms_qkv_itype = RmsQkvRopeAppend(n=HIDDEN_DIM, head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS)
    _rms_upgate_silu_itype = RmsUpgateSilu(n=HIDDEN_DIM)
    _rms_lm_head_itype = RmsLmHead(n=HIDDEN_DIM)
    instruction_metas = [
        _noop_inst_meta,
        InstructionMeta(icode=ICODE_QKV, itype=_rms_qkv_itype,
                        src_tensors=(T.HIDDEN_STATES, T.ATTN_NORM_WEIGHTS,
                                     T.QKV_WEIGHTS, T.ROPE_COS, T.ROPE_SIN,
                                     T.K_CACHE, T.V_CACHE),
                        dst_tensors=(T.Q_POST_ROPE,)),
        InstructionMeta(icode=ICODE_ATTENTION, itype=_attention_partial_itype,
                        src_tensors=(T.Q_POST_ROPE, T.K_CACHE, T.V_CACHE),
                        dst_tensors=(T.ATTN_OUT,)),
        InstructionMeta(icode=ICODE_O_PROJ, itype=_matvec_adds_itype,
                        src_tensors=(T.ATTN_OUT, T.O_WEIGHTS),
                        dst_tensors=(T.HIDDEN_STATES,)),
        InstructionMeta(icode=ICODE_UPGATE, itype=_rms_upgate_silu_itype,
                        src_tensors=(T.HIDDEN_STATES, T.MLP_NORM_WEIGHTS,
                                     T.UP_WEIGHTS, T.GATE_WEIGHTS),
                        dst_tensors=(T.SILU_OUT,)),
        InstructionMeta(icode=ICODE_DOWNPROJ, itype=_matvec_adds_itype,
                        src_tensors=(T.SILU_OUT, T.DOWN_WEIGHTS),
                        dst_tensors=(T.HIDDEN_STATES,)),
        InstructionMeta(icode=ICODE_LM_HEAD, itype=_rms_lm_head_itype,
                        src_tensors=(T.HIDDEN_STATES, T.LM_HEAD_NORM_WEIGHT,
                                     T.LM_HEAD_WEIGHT),
                        dst_tensors=(T.LOGITS,)),
    ]

    instructions: list[Instruction] = []
    num_barriers = num_layers * NUM_BARRIER_SLOTS * MAX_SUB_BARRIERS

    for layer_idx in range(num_layers):
        prev_layer_barrier = _barrier_index(layer_idx - 1, 5, 0) if layer_idx > 0 else None
        prev_layer_target = DOWNPROJ_TARGET if layer_idx > 0 else 0

        qkv_insts = _schedule_qkv(
            sm_count, layer_idx, icode_qkv,
            prev_layer_barrier, prev_layer_target,
        )
        attn_insts = _schedule_attention(layer_idx, icode_attn)
        oproj_insts = _schedule_o_proj(sm_count, layer_idx, icode_oproj)
        upgate_insts = _schedule_upgate(sm_count, layer_idx, icode_upgate)
        downproj_insts = _schedule_downproj(sm_count, layer_idx, icode_downproj)

        instructions.extend(qkv_insts)
        instructions.extend(attn_insts)
        instructions.extend(oproj_insts)
        instructions.extend(upgate_insts)
        instructions.extend(downproj_insts)

    # LM head
    lmhead_insts = _schedule_lm_head(sm_count, num_layers, icode_lmhead)
    instructions.extend(lmhead_insts)

    # All tensors are inputs (caller pre-allocates everything)
    input_tensor_indices = tuple(range(T.COUNT))
    output_tensor_indices = (T.LOGITS,)

    return (
        instruction_metas,
        tensor_metas,
        instructions,
        num_barriers,
        input_tensor_indices,
        output_tensor_indices,
    )
