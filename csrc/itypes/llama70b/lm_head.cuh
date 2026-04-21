#pragma once

#include "kittens.cuh"

namespace megakittens {

// LMHead — LLaMA-70B final logit projection (reference opcode 11,
// matching `lm_head.cu::lm_head_op`).
//
// Reference semantics:
// - bf16 GEMM: logits = hidden_states @ lm_head_weights, where
//   hidden_states is already RMS-normed (produced by LM_HeadNorm).
// - Uses the same `matmul_pipeline` infrastructure as other matmuls,
//   but stores bf16 logits tiles (rt_bf<16, 256>) via a dedicated
//   tile -> g.logits TMA path, with a barrier at
//   `10 + (instruction_index % 2)` handshaking consumer -> storer.
// - Shape: hidden_dim -> vocab_size (e.g. 8192 -> 128k).
//
// STATUS: not ported as a distinct itype. The current port piggybacks
// on the generic `Gemm` in csrc/itypes/gemm.cuh (PORT_STATUS: "LM_Head
// -> existing Gemm — plain matmul, no fusion"). A faithful port would
// add the dedicated logits TMA store path and the (10 + inst%2) barrier
// pattern; skipping that loses the final-layer storer/consumer overlap
// but is otherwise correct.
//
// See csrc/itypes/reference/lm_head.cu for the reference.

// TODO if the final-layer consumer/storer overlap is worth the code.

}  // namespace megakittens
