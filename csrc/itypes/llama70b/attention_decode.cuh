#pragma once

#include "kittens.cuh"

namespace megakittens {

// AttentionDecode — LLaMA-70B GQA decode attention (reference opcode 4,
// matching `attention_decode.cu::gqa_attention_decode_op`).
//
// Reference semantics:
// - 8 sequences per instruction, 3-stage KV pipeline. Nested loop:
//     outer over pages, inner 8 iterations per page (iters_per_page =
//     kv_page_size / decode_kv_block_size = 128 / 16 = 8).
// - Q loaded as 8-row masked warp load (GQA_RATIO=8 Q heads per KV head).
// - Softmax in log-space with sequence-length masking on the tail block:
//     (col >= remaining_length) ? -inf : val.
// - Output: 16x16 per-sequence tile, atomic-stored to smem, reduced
//   across warps, TMA-stored to attn_out.
//
// Int32 global reads required (the blocker):
//   decode_kv_indptr   : gl<int, 1, 1, 1, -1>
//   decode_kv_indices  : gl<int, 1, 1, 1, -1>
//   sequence_length    : gl<int, 1, 1, 1, -1>   — per-sequence, for masking
//
// Access pattern:
//   indptr_start = decode_kv_indptr[global_seq_idx];
//   num_pages    = decode_kv_indptr[global_seq_idx+1] - indptr_start;
//   kv_page      = decode_kv_indices[indptr_start + i/iters_per_page];
//
// STATUS: BLOCKED on int32 gl read (see PORT_STATUS.md). Same as prefill.
//
// See csrc/itypes/reference/attention_decode.cu for the reference.

// TODO once int32 is unblocked.

}  // namespace megakittens
