// Entry point for the manual-kernel ablation suite. Each kernel lives in its
// own .cuh; this file collects them under a single PYBIND11_MODULE so they
// build into one `_C.so` that manual_decode.py imports.

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include "rms.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_forward", &manual_kernels::rms_dispatch,
          "RMSNorm forward (bf16). out = x * rsqrt(mean(x^2) + eps) * weight. "
          "Args: x [B, 8192] bf16, weight [8192] bf16, eps [1] fp32, out [B, 8192] bf16 (mutated).");
}
