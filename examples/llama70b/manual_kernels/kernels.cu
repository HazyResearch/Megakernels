// Entry point for the manual-kernel ablation suite. Each kernel lives in its
// own .cuh; this file collects them under a single PYBIND11_MODULE so they
// build into one `_C.so` that manual_decode.py imports.

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include "rms.cuh"
#include "qkv_rope_append.cuh"
#include "gate_silu.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_forward", &manual_kernels::rms_dispatch,
          "RMSNorm forward (bf16). out = x * rsqrt(mean(x^2) + eps) * weight. "
          "Args: x [B, 8192] bf16, weight [8192] bf16, eps [1] fp32, out [B, 8192] bf16 (mutated).");
    m.def("qkv_rope_append_forward", &manual_kernels::qkv_rope_append_dispatch,
          "Fused QKV proj + RoPE on Q/K + scatter K/V into paged cache. "
          "Args: x [B, 8192] bf16, qkv_w [1, 10240, 8192] bf16 (Q/K rows pre-interleaved), "
          "rope_cos/rope_sin [S, 128] fp32 (interleaved), pos_id [1] i32, append_ids [B] i32, "
          "k_cache/v_cache [pages, 128, 8, 128] bf16 (layer slice, mutated), q_out [B, 8192] bf16.");
    m.def("gate_silu_forward", &manual_kernels::gate_silu_dispatch,
          "Gate-SiLU matmul: out = silu(x @ gate_w[0].T). "
          "Args: x [M, K] bf16, gate_w [1, N, K] bf16, out [M, N] bf16. "
          "M%512==0, N%256==0, K%64==0.");
}
