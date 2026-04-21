#pragma once

#include "kittens.cuh"

namespace megakittens {

// QKVRopeAppend — LLaMA-70B fused QKV projection + RoPE + KV cache append
// (reference opcode 2, matching `qkv_rope_append.cu::qkv_rope_append_op`).
//
// Reference semantics:
// - Fused matmul: [Q|K|V] = hidden_states @ [W_q|W_k|W_v].
// - RoPE applied to Q (all heads) and K (kv heads only) via cos/sin
//   vectors gathered per token by `position_ids[token]`.
// - K/V tiles TMA-store-async into the paged KV cache at
//     (num_pages * layer + append_idx / page_size,
//      append_idx % page_size, kv_head, 0)
//   with `append_idx = append_indices[token]`.
// - BSHD layout (B, S, H, D). 8-way TP: each device owns
//   num_heads/8 Q heads and num_kv_heads/8 = 1 KV head.
//
// Int32 global reads required (these are the blocker):
//   position_ids    : gl<int, 1, 1, 1, -1>  — per-token RoPE angle index
//   append_indices  : gl<int, 1, 1, 1, -1>  — per-token KV cache slot
//
// STATUS: stub — int32 gl read fix pattern known (use `warp::load` into a sv<int, N> scratch,
//         as rope_gather.cuh does; verified working on 8×B200). The bf16-only `Rope` itype
//         (rope.cuh) ports the RoPE math on pre-gathered cos/sin; this itype fuses
//         matmul + RoPE + KV-append around that and still needs its kernel body written.
//
// See PORT_STATUS.md section "The int32 blocker" and the reference at
// csrc/itypes/reference/qkv_rope_append.cu.

// TODO once int32 is unblocked — template signature mirrors the reference:
// template <typename Config, typename Globals,
//           int SRC_HIDDEN_PGL, int SRC_WQ, int SRC_WK, int SRC_WV,
//           int SRC_COS, int SRC_SIN, int SRC_POS_IDS, int SRC_APPEND_IDS,
//           int DST_Q, int DST_KCACHE, int DST_VCACHE>
// struct QKVRopeAppend { ... controller / loader / launcher / consumer / storer ... };

}  // namespace megakittens
