#pragma once

#include "kittens.cuh"

namespace megakittens {

// AttentionPrefill — LLaMA-70B GQA prefill attention (reference opcode 3,
// matching `attention_prefill.cu::gqa_attention_prefill_op`).
//
// Reference semantics:
// - Full-sequence FlashAttention tiling: 16x128 Q tiles, KV in 128-row
//   pages, 2-stage KV load pipeline.
// - Softmax in log-space: row_max / sub / exp2 / sum, with causal masking
//   per block (`q_pos_start = prefill_token_offset + rel_q_row`).
// - Output reduction via dynamic logsumexp scaling across blocks.
// - GQA: 8 Q heads per KV head. Q is reshaped from 16x512 to 64x128 tiles
//   for WGMMA alignment.
// - Shapes (hardcoded in reference): num_heads=64, num_kv_heads=8,
//   head_dim=128, kv_page_size=128, GQA_RATIO=8.
//
// Int32 global reads required (the blocker):
//   prefill_qo_indptr  : gl<int, 1, 1, 1, -1>  — [seq_start, seq_end) for Q
//   prefill_kv_indptr  : gl<int, 1, 1, 1, -1>  — kv_indices offset per seq
//   prefill_kv_indices : gl<int, 1, 1, 1, -1>  — physical page index per block
//
// Access pattern:
//   kv_indptr_start = prefill_kv_indptr[seq_idx];
//   kv_page         = prefill_kv_indices[kv_indptr_start + block_idx];
//   tma::load_async(K/V, {num_pages*layer + kv_page, 0, kv_head, 0}, ...)
//
// STATUS: BLOCKED on int32 gl read (see PORT_STATUS.md). The framework's
// generic `Attention` in csrc/itypes/attention.cuh is NOT a faithful
// substitute — it lacks paged KV, indptr indirection, and prefill-offset
// causal masking.
//
// See csrc/itypes/reference/attention_prefill.cu for the reference.

// TODO once int32 is unblocked.

}  // namespace megakittens
