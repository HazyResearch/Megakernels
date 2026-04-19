"""
Llama-1B decode scheduler.

Produces a flat instruction list with barrier dependencies,
matching the megakittens Instruction/InstructionMeta format.
"""

from __future__ import annotations

from megakittens.itypes.llama1b.attention_partial import AttentionPartial, AttentionPartialMulti
from megakittens.itypes.llama1b.attention_reduction import AttentionReduction
from megakittens.itypes.noop import Noop
from megakittens.itypes.llama1b.matvec_adds import MatVecAdds
from megakittens.itypes.llama1b.rms_lm_head import RmsLmHead
from megakittens.itypes.llama1b.rms_qkv_rope_append import RmsQkvRopeAppend
from megakittens.itypes.llama1b.rms_upgate_silu import RmsUpgateSilu
from megakittens.schema.device import Device
from megakittens.schema.dtype import DType
from megakittens.schema.instruction import Instruction, InstructionMeta
from megakittens.schema.tensor import TensorMeta


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
MAX_SEQ_LEN = 4096

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
    POS_ID = 18              # [1] int32
    ATTN_SCALE = 19          # [1] fp32
    RMS_NORM_EPS = 20        # [1] fp32
    ATTN_O_INTER = 21        # [NUM_ATTENTION_HEADS, num_partitions, HEAD_DIM] fp32
    ATTN_L_INTER = 22        # [NUM_ATTENTION_HEADS, num_partitions] fp32
_TENSOR_COUNT = max(v for k, v in vars(T).items() if not k.startswith('_') and isinstance(v, int)) + 1
T.COUNT = _TENSOR_COUNT

CLUSTER_SIZE = 2
GQA_RATIO = NUM_ATTENTION_HEADS // NUM_KV_HEADS  # 4
BLOCKS_PER_HEAD = HEAD_DIM // MATVEC_BLOCK_SIZE   # 4

# Barrier layout: barriers[layer][opcode_slot][sub_idx]
NUM_BARRIER_SLOTS = 6   # qkv, attn, attn_red, oproj, upgate, downproj
MAX_SUB_BARRIERS = 48   # largest: QKV with 32 Q + 8 K + 8 V = 48

# Sub-barrier counts per opcode slot
# QKV layout matches rms_qkv_rope_append.cuh::subregion_offset: Q per kv_head_group, K/V per kv_head
QKV_SUB_BARRIERS = 3 * NUM_KV_HEADS                         # 24
ATTN_SUB_BARRIERS = NUM_ATTENTION_HEADS                     # 32
ATTN_RED_SUB_BARRIERS = 1
OPROJ_SUB_BARRIERS = 1
UPGATE_SUB_BARRIERS = INTERMEDIATE_DIM // HIDDEN_DIM        # 4
DOWNPROJ_SUB_BARRIERS = 1

# Target counts for fine-grained barriers
UPGATE_TARGET_PER_SUB = HIDDEN_DIM // MATVEC_BLOCK_SIZE                 # 128
ATTN_RED_TARGET = NUM_KV_HEADS                                          # 8 (one arrive per attention inst)
# OPROJ/DOWNPROJ targets = number of aggregated instructions (one arrive per instance).


# mirror of rms_qkv_rope_append.cuh::subregion_offset
def _qkv_subregion(block_idx: int) -> int:
    k_blk_start = Q_DIM // MATVEC_BLOCK_SIZE
    v_blk_start = (Q_DIM + NUM_KV_HEADS * HEAD_DIM) // MATVEC_BLOCK_SIZE
    blocks_per_head = HEAD_DIM // MATVEC_BLOCK_SIZE
    blocks_per_group = blocks_per_head * GQA_RATIO
    if block_idx < k_blk_start:
        return block_idx // blocks_per_group
    if block_idx < v_blk_start:
        return NUM_KV_HEADS + (block_idx - k_blk_start) // blocks_per_head
    return 2 * NUM_KV_HEADS + (block_idx - v_blk_start) // blocks_per_head


def _barrier_index(layer_idx: int, opcode_slot: int, sub_idx: int = 0) -> int:
    return layer_idx * NUM_BARRIER_SLOTS * MAX_SUB_BARRIERS + opcode_slot * MAX_SUB_BARRIERS + sub_idx


_noop_inst_meta = InstructionMeta(icode=0, itype=Noop(), src_tensors=(), dst_tensors=())

_noop = Instruction(
    icode=0, src_tensors=(), dst_tensors=(), indices=(),
    src_barriers=(), src_barrier_targets=(),
    num_src_input_barriers=0, num_src_reuse_barriers=0, num_dst_input_barriers=0, num_dst_reuse_barriers=0,
    dst_barriers=(),
)


def _pad_to_cluster(instructions: list[Instruction]) -> None:
    while len(instructions) % CLUSTER_SIZE != 0:
        instructions.append(_noop)

def _make_tensor_metas(device: Device, num_partitions: int = 1) -> list[TensorMeta]:
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
    metas[T.POS_ID] = TensorMeta(dtype=DType.int32, shape=(1,), device=device)
    metas[T.ATTN_SCALE] = TensorMeta(dtype=fp32, shape=(1,), device=device)
    metas[T.RMS_NORM_EPS] = TensorMeta(dtype=fp32, shape=(1,), device=device)
    metas[T.ATTN_O_INTER] = TensorMeta(dtype=fp32, shape=(NUM_ATTENTION_HEADS, num_partitions, HEAD_DIM), device=device)
    # Pad L columns to multiple of 16 for TMA sv alignment
    l_cols = ((num_partitions + 15) // 16) * 16
    metas[T.ATTN_L_INTER] = TensorMeta(dtype=fp32, shape=(NUM_ATTENTION_HEADS, l_cols), device=device)

    return metas


# -- Per-instruction-group schedulers -----------------------------------------

def _schedule_qkv(
    sm_count: int,
    layer_idx: int,
    icode: int,
    prev_barrier: int | None,
    prev_barrier_target: int,
) -> tuple[list[Instruction], list[int]]:
    # target_per_sub[s] = number of chunks that arrive on sub-barrier s
    num_blocks = QKV_DIM // MATVEC_BLOCK_SIZE  # 192
    chunks = []
    for sm in range(sm_count):
        start = round(sm * num_blocks / sm_count)
        end = round((sm + 1) * num_blocks / sm_count)
        if start == end:
            continue
        subs = []
        prev_sub = -1
        for b in range(start, end):
            s = _qkv_subregion(b)
            if s != prev_sub:
                subs.append(s)
                prev_sub = s
        chunks.append((start, end, subs))

    target_per_sub = [0] * QKV_SUB_BARRIERS
    for _, _, subs in chunks:
        for s in subs:
            target_per_sub[s] += 1

    instructions = []
    for start, end, subs in chunks:
        dst_barriers = tuple(_barrier_index(layer_idx, 0, s) for s in subs)
        src_barriers = (prev_barrier,) if prev_barrier is not None else ()
        src_targets = (prev_barrier_target,) if prev_barrier is not None else ()
        instructions.append(Instruction(
            icode=icode,
            src_tensors=(T.HIDDEN_STATES, T.ATTN_NORM_WEIGHTS, T.QKV_WEIGHTS,
                         T.ROPE_COS, T.ROPE_SIN, T.K_CACHE, T.V_CACHE),
            dst_tensors=(T.Q_POST_ROPE, T.K_CACHE, T.V_CACHE),
            indices=(layer_idx, start, end),
            src_barriers=src_barriers,
            src_barrier_targets=src_targets,
            num_src_input_barriers=len(src_barriers),
            num_src_reuse_barriers=0,
            num_dst_input_barriers=len(dst_barriers),
            num_dst_reuse_barriers=0,
            dst_barriers=dst_barriers,
        ))
    _pad_to_cluster(instructions)
    return instructions, target_per_sub


def _schedule_attention(
    layer_idx: int,
    icode: int,
    qkv_target_per_sub: list[int],
) -> list[Instruction]:
    # No attention_reduction step currently (1 partial per head),
    # so attention signals the attn_red barrier (slot 2) directly.
    attn_red_barrier = _barrier_index(layer_idx, 2, 0)
    instructions = []
    for kv_head in range(NUM_KV_HEADS):
        q_sub = kv_head
        k_sub = NUM_KV_HEADS + kv_head
        v_sub = 2 * NUM_KV_HEADS + kv_head
        src_barriers = (
            _barrier_index(layer_idx, 0, q_sub),
            _barrier_index(layer_idx, 0, k_sub),
            _barrier_index(layer_idx, 0, v_sub),
        )
        src_targets = (
            qkv_target_per_sub[q_sub],
            qkv_target_per_sub[k_sub],
            qkv_target_per_sub[v_sub],
        )
        instructions.append(Instruction(
            icode=icode,
            src_tensors=(T.Q_POST_ROPE, T.K_CACHE, T.V_CACHE),
            dst_tensors=(T.ATTN_OUT,),
            indices=(layer_idx, kv_head),
            src_barriers=src_barriers,
            src_barrier_targets=src_targets,
            num_src_input_barriers=len(src_barriers),
            num_src_reuse_barriers=0,
            num_dst_input_barriers=1,
            num_dst_reuse_barriers=0,
            dst_barriers=(attn_red_barrier,),
        ))
    _pad_to_cluster(instructions)
    return instructions


def _schedule_attention_multi(
    layer_idx: int,
    icode: int,
    num_partitions: int,
    qkv_target_per_sub: list[int],
) -> list[Instruction]:
    instructions = []
    for kv_head in range(NUM_KV_HEADS):
        q_sub = kv_head
        k_sub = NUM_KV_HEADS + kv_head
        v_sub = 2 * NUM_KV_HEADS + kv_head
        src_barriers = (
            _barrier_index(layer_idx, 0, q_sub),
            _barrier_index(layer_idx, 0, k_sub),
            _barrier_index(layer_idx, 0, v_sub),
        )
        src_targets = (
            qkv_target_per_sub[q_sub],
            qkv_target_per_sub[k_sub],
            qkv_target_per_sub[v_sub],
        )
        attn_barrier = _barrier_index(layer_idx, 1, kv_head)
        for partial_idx in range(num_partitions):
            instructions.append(Instruction(
                icode=icode,
                src_tensors=(T.Q_POST_ROPE, T.K_CACHE, T.V_CACHE),
                dst_tensors=(T.ATTN_O_INTER, T.ATTN_L_INTER),
                indices=(layer_idx, kv_head, partial_idx, num_partitions, attn_barrier),
                src_barriers=src_barriers,
                src_barrier_targets=src_targets,
                num_src_input_barriers=len(src_barriers),
                num_src_reuse_barriers=0,
                num_dst_input_barriers=0,
                num_dst_reuse_barriers=0,
                dst_barriers=(),
            ))
    _pad_to_cluster(instructions)
    return instructions


def _schedule_attention_reduction(
    layer_idx: int,
    icode: int,
    num_partitions: int,
) -> list[Instruction]:
    attn_red_barrier = _barrier_index(layer_idx, 2, 0)
    instructions = []
    for kv_head in range(NUM_KV_HEADS):
        q_head_start = kv_head * GQA_RATIO
        attn_barrier = _barrier_index(layer_idx, 1, kv_head)
        instructions.append(Instruction(
            icode=icode,
            src_tensors=(T.ATTN_L_INTER, T.ATTN_O_INTER),
            dst_tensors=(T.ATTN_OUT,),
            indices=(layer_idx, q_head_start, num_partitions, attn_red_barrier),
            src_barriers=(attn_barrier,),
            src_barrier_targets=(num_partitions * GQA_RATIO,),
            num_src_input_barriers=1,
            num_src_reuse_barriers=0,
            num_dst_input_barriers=0,
            num_dst_reuse_barriers=0,
            dst_barriers=(),
        ))
    _pad_to_cluster(instructions)
    return instructions


def _schedule_o_proj(
    sm_count: int,
    layer_idx: int,
    icode: int,
    max_instructions: int | None = None,
) -> tuple[list[Instruction], int]:
    attn_red_barrier = _barrier_index(layer_idx, 2, 0)
    oproj_barrier = _barrier_index(layer_idx, 3, 0)
    num_blocks = HIDDEN_DIM // MATVEC_BLOCK_SIZE  # 128
    num_instructions = min(sm_count, num_blocks)
    if max_instructions is not None:
        num_instructions = min(num_instructions, max_instructions)
    instructions = []
    for i in range(num_instructions):
        start = round(i * num_blocks / num_instructions)
        end = round((i + 1) * num_blocks / num_instructions)
        if start == end:
            continue
        instructions.append(Instruction(
            icode=icode,
            src_tensors=(T.HIDDEN_STATES, T.ATTN_OUT, T.O_WEIGHTS),
            dst_tensors=(T.HIDDEN_STATES,),
            indices=(layer_idx, start, end, 0),
            src_barriers=(attn_red_barrier,),
            src_barrier_targets=(ATTN_RED_TARGET,),
            num_src_input_barriers=1,
            num_src_reuse_barriers=0,
            num_dst_input_barriers=1,
            num_dst_reuse_barriers=0,
            dst_barriers=(oproj_barrier,),
        ))
    target = len(instructions)
    _pad_to_cluster(instructions)
    return instructions, target


def _schedule_upgate(
    sm_count: int,
    layer_idx: int,
    icode: int,
    oproj_target: int,
) -> list[Instruction]:
    oproj_barrier = _barrier_index(layer_idx, 3, 0)
    num_blocks = INTERMEDIATE_DIM // MATVEC_BLOCK_SIZE  # 512
    blocks_per_sub = HIDDEN_DIM // MATVEC_BLOCK_SIZE    # 128
    instructions = []
    for sm in range(sm_count):
        # one sub-barrier per chunk this SM's gate iters touch (sm + ii * sm_count)
        dst_barriers = tuple(
            _barrier_index(layer_idx, 4, (sm + ii * sm_count) // blocks_per_sub)
            for ii in range((num_blocks - sm + sm_count - 1) // sm_count)
        )
        instructions.append(Instruction(
            icode=icode,
            src_tensors=(T.HIDDEN_STATES, T.MLP_NORM_WEIGHTS, T.UP_WEIGHTS, T.GATE_WEIGHTS),
            dst_tensors=(T.SILU_OUT,),
            indices=(layer_idx, sm, sm_count, num_blocks),
            src_barriers=(oproj_barrier,),
            src_barrier_targets=(oproj_target,),
            num_src_input_barriers=1,
            num_src_reuse_barriers=0,
            num_dst_input_barriers=len(dst_barriers),
            num_dst_reuse_barriers=0,
            dst_barriers=dst_barriers,
        ))
    _pad_to_cluster(instructions)
    return instructions


def _schedule_downproj(
    sm_count: int,
    layer_idx: int,
    icode: int,
) -> tuple[list[Instruction], int]:
    downproj_barrier = _barrier_index(layer_idx, 5, 0)
    num_blocks = HIDDEN_DIM // MATVEC_BLOCK_SIZE  # 128
    num_chunks = INTERMEDIATE_DIM // HIDDEN_DIM   # 4 reduction chunks
    sms_per_chunk = sm_count // num_chunks

    instructions = []
    for chunk in range(num_chunks):
        upgate_sub = _barrier_index(layer_idx, 4, chunk)
        col_offset = chunk * HIDDEN_DIM
        # Distribute blocks within this chunk across its SMs
        for sm in range(sms_per_chunk):
            start = round(sm * num_blocks / sms_per_chunk)
            end = round((sm + 1) * num_blocks / sms_per_chunk)
            if start == end:
                continue
            instructions.append(Instruction(
                icode=icode,
                src_tensors=(T.HIDDEN_STATES, T.SILU_OUT, T.DOWN_WEIGHTS),
                dst_tensors=(T.HIDDEN_STATES,),
                indices=(layer_idx, start, end, col_offset),
                src_barriers=(upgate_sub,),
                src_barrier_targets=(UPGATE_TARGET_PER_SUB,),
                num_src_input_barriers=1,
                num_src_reuse_barriers=0,
                num_dst_input_barriers=1,
                num_dst_reuse_barriers=0,
                dst_barriers=(downproj_barrier,),
            ))

    target = len(instructions)
    _pad_to_cluster(instructions)
    return instructions, target


def _schedule_lm_head(
    sm_count: int,
    num_layers: int,
    icode: int,
    prev_barrier_target: int,
) -> list[Instruction]:
    prev_barrier = _barrier_index(num_layers - 1, 5, 0)
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
            num_src_input_barriers=1,
            num_src_reuse_barriers=0,
            num_dst_input_barriers=0,
            num_dst_reuse_barriers=0,
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
ICODE_ATTENTION_MULTI = 7
ICODE_ATTENTION_RED = 8


def schedule_decode(
    sm_count: int,
    num_layers: int = NUM_LAYERS,
    device: Device | None = None,
    noop: bool = False,
    oproj_max_instructions: int | None = 64,
    num_partitions: int = 1,
    max_partitions: int | None = None,
):
    if device is None:
        device = Device(type="cuda", index=0)

    if max_partitions is None:
        max_partitions = num_partitions

    use_multi_partition = num_partitions > 1

    tensor_metas = _make_tensor_metas(device, num_partitions=max_partitions)

    # Icode mapping — noop mode sets everything to 0
    icode_qkv = 0 if noop else ICODE_QKV
    icode_attn = 0 if noop else ICODE_ATTENTION
    icode_attn_multi = 0 if noop else ICODE_ATTENTION_MULTI
    icode_attn_red = 0 if noop else ICODE_ATTENTION_RED
    icode_oproj = 0 if noop else ICODE_O_PROJ
    icode_upgate = 0 if noop else ICODE_UPGATE
    icode_downproj = 0 if noop else ICODE_DOWNPROJ
    icode_lmhead = 0 if noop else ICODE_LM_HEAD

    # Instruction metas (drive JIT codegen)
    _matvec_adds_itype = MatVecAdds(n=HIDDEN_DIM)
    _attention_partial_itype = AttentionPartial()
    _attention_partial_multi_itype = AttentionPartialMulti()
    _attention_reduction_itype = AttentionReduction(
        head_dim=HEAD_DIM, q_heads_per_instruction=GQA_RATIO,
        max_partials=max_partitions,
    )
    _rms_qkv_itype = RmsQkvRopeAppend(n=HIDDEN_DIM, head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS)
    _rms_upgate_silu_itype = RmsUpgateSilu(n=HIDDEN_DIM)
    _rms_lm_head_itype = RmsLmHead(n=HIDDEN_DIM)
    instruction_metas = [
        _noop_inst_meta,
        InstructionMeta(icode=ICODE_QKV, itype=_rms_qkv_itype,
                        src_tensors=(T.HIDDEN_STATES, T.ATTN_NORM_WEIGHTS,
                                     T.QKV_WEIGHTS, T.ROPE_COS, T.ROPE_SIN,
                                     T.K_CACHE, T.V_CACHE,
                                     T.POS_ID, T.RMS_NORM_EPS),
                        dst_tensors=(T.Q_POST_ROPE, T.K_CACHE, T.V_CACHE)),
        InstructionMeta(icode=ICODE_O_PROJ, itype=_matvec_adds_itype,
                        src_tensors=(T.HIDDEN_STATES, T.ATTN_OUT, T.O_WEIGHTS),
                        dst_tensors=(T.HIDDEN_STATES,)),
        InstructionMeta(icode=ICODE_UPGATE, itype=_rms_upgate_silu_itype,
                        src_tensors=(T.HIDDEN_STATES, T.MLP_NORM_WEIGHTS,
                                     T.UP_WEIGHTS, T.GATE_WEIGHTS,
                                     T.RMS_NORM_EPS),
                        dst_tensors=(T.SILU_OUT,)),
        InstructionMeta(icode=ICODE_DOWNPROJ, itype=_matvec_adds_itype,
                        src_tensors=(T.HIDDEN_STATES, T.SILU_OUT, T.DOWN_WEIGHTS),
                        dst_tensors=(T.HIDDEN_STATES,)),
        InstructionMeta(icode=ICODE_LM_HEAD, itype=_rms_lm_head_itype,
                        src_tensors=(T.HIDDEN_STATES, T.LM_HEAD_NORM_WEIGHT,
                                     T.LM_HEAD_WEIGHT, T.RMS_NORM_EPS),
                        dst_tensors=(T.LOGITS,)),
    ]

    if max_partitions > 1:
        # Include all attention icodes so one compiled kernel supports both paths
        instruction_metas += [
            InstructionMeta(icode=ICODE_ATTENTION, itype=_attention_partial_itype,
                            src_tensors=(T.Q_POST_ROPE, T.K_CACHE, T.V_CACHE,
                                         T.POS_ID, T.ATTN_SCALE),
                            dst_tensors=(T.ATTN_OUT,)),
            InstructionMeta(icode=ICODE_ATTENTION_MULTI, itype=_attention_partial_multi_itype,
                            src_tensors=(T.Q_POST_ROPE, T.K_CACHE, T.V_CACHE,
                                         T.POS_ID, T.ATTN_SCALE),
                            dst_tensors=(T.ATTN_O_INTER, T.ATTN_L_INTER)),
            InstructionMeta(icode=ICODE_ATTENTION_RED, itype=_attention_reduction_itype,
                            src_tensors=(T.ATTN_L_INTER, T.ATTN_O_INTER),
                            dst_tensors=(T.ATTN_OUT,)),
        ]
    else:
        instruction_metas.append(
            InstructionMeta(icode=ICODE_ATTENTION, itype=_attention_partial_itype,
                            src_tensors=(T.Q_POST_ROPE, T.K_CACHE, T.V_CACHE,
                                         T.POS_ID, T.ATTN_SCALE),
                            dst_tensors=(T.ATTN_OUT,)),
        )

    instructions: list[Instruction] = []
    num_barriers = num_layers * NUM_BARRIER_SLOTS * MAX_SUB_BARRIERS

    prev_downproj_target = 0
    prev_layer_barrier = None

    for layer_idx in range(num_layers):
        qkv_insts, qkv_target_per_sub = _schedule_qkv(
            sm_count, layer_idx, icode_qkv,
            prev_layer_barrier, prev_downproj_target,
        )

        if use_multi_partition:
            attn_insts = _schedule_attention_multi(
                layer_idx, icode_attn_multi, num_partitions, qkv_target_per_sub)
            attn_red_insts = _schedule_attention_reduction(
                layer_idx, icode_attn_red, num_partitions)
        else:
            attn_insts = _schedule_attention(layer_idx, icode_attn, qkv_target_per_sub)
            attn_red_insts = []

        oproj_insts, oproj_target = _schedule_o_proj(
            sm_count, layer_idx, icode_oproj,
            max_instructions=oproj_max_instructions,
        )
        upgate_insts = _schedule_upgate(sm_count, layer_idx, icode_upgate, oproj_target)
        downproj_insts, downproj_target = _schedule_downproj(
            sm_count, layer_idx, icode_downproj,
        )

        instructions.extend(qkv_insts)
        instructions.extend(attn_insts)
        instructions.extend(attn_red_insts)
        instructions.extend(oproj_insts)
        instructions.extend(upgate_insts)
        instructions.extend(downproj_insts)

        prev_layer_barrier = _barrier_index(layer_idx, 5, 0)
        prev_downproj_target = downproj_target

    # LM head
    lmhead_insts = _schedule_lm_head(sm_count, num_layers, icode_lmhead, prev_downproj_target)
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
