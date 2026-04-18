"""Dump both the hand-written and auto-compiled llama1b decode schedules.

Produces:
  log/hand_decode.NN.schedule.txt   (from examples.llama1b.scheduler.schedule_decode)
  log/decode.NN.schedule.txt        (from megakittens.compile(compiled_decode.decode))

No HF weights needed; uses zero-filled tensors with correct shapes/dtypes.
"""

from __future__ import annotations

import torch

import megakittens
from megakittens.jit.cuda_utils import get_sm_count, initialize_cuda_context
from megakittens.utils import create_log_base_path, save_schedule_as_txt

from .compiled_decode import decode
from .scheduler import (
    HEAD_DIM, HIDDEN_DIM, INTERMEDIATE_DIM, MAX_SEQ_LEN, NUM_KV_HEADS,
    NUM_LAYERS, Q_DIM, QKV_DIM, VOCAB_SIZE, schedule_decode,
)


def dump_hand(sm_count: int) -> None:
    instruction_metas, tensor_metas, instructions, num_barriers, _, _ = schedule_decode(
        sm_count=sm_count, num_partitions=1,
    )
    base = create_log_base_path(fn=type("hand_decode", (), {"__qualname__": "hand_decode"}))
    save_schedule_as_txt(tensor_metas, instructions, instruction_metas, num_barriers, base)
    print(f"[hand] instructions={len(instructions)}  barriers={num_barriers}  -> {base}.schedule.txt")


def dump_auto() -> None:
    D = "cuda"
    bf16 = torch.bfloat16
    fp32 = torch.float32

    # tensors matching the `decode` signature in compiled_decode.py
    args = (
        torch.zeros(HIDDEN_DIM, dtype=bf16, device=D),                                       # hidden_states
        torch.zeros(NUM_LAYERS, QKV_DIM, HIDDEN_DIM, dtype=bf16, device=D),                  # qkv_weights
        torch.zeros(NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM, dtype=bf16, device=D),               # o_weights
        torch.zeros(NUM_LAYERS, HIDDEN_DIM, dtype=bf16, device=D),                           # attn_norm
        torch.zeros(NUM_LAYERS, HIDDEN_DIM, dtype=bf16, device=D),                           # mlp_norm
        torch.zeros(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=bf16, device=D),         # up
        torch.zeros(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=bf16, device=D),         # gate
        torch.zeros(NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM, dtype=bf16, device=D),         # down
        torch.zeros(HIDDEN_DIM, dtype=bf16, device=D),                                       # lm_head_norm
        torch.zeros(VOCAB_SIZE, HIDDEN_DIM, dtype=bf16, device=D),                           # lm_head
        torch.zeros(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=bf16, device=D),  # k_cache
        torch.zeros(NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=bf16, device=D),  # v_cache
        torch.zeros(MAX_SEQ_LEN, HEAD_DIM, dtype=fp32, device=D),                            # rope_cos
        torch.zeros(MAX_SEQ_LEN, HEAD_DIM, dtype=fp32, device=D),                            # rope_sin
        torch.tensor([1], dtype=torch.int32, device=D),                                      # pos_id
        torch.tensor([1.0], dtype=fp32, device=D),                                           # attn_scale
        torch.tensor([1e-5], dtype=fp32, device=D),                                          # rms_norm_eps
    )
    compiled = megakittens.compile(
        decode, use_jit_cache=False, verbose=False, save_schedule=True, dry_run=False,
    )
    with torch.inference_mode():
        compiled(*args)
    print("[auto] compile done; schedule dumped to log/decode.NN.schedule.txt")


if __name__ == "__main__":
    initialize_cuda_context()
    torch._dynamo.reset()
    sm = get_sm_count()
    print(f"sm_count={sm}")
    dump_hand(sm)
    dump_auto()
