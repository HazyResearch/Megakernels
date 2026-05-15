#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

namespace manual_kernels {

// Mirrors the megakernel's rms70b work split. At batch B, we launch
// `rms_n_inst(B)` blocks; each block handles a contiguous slice of rows.
// At B = 1024 this is 148 blocks (matching the megakernel's 148 instructions
// at B=1024 on B200), giving instructions:blocks = 1:1.

constexpr int RMS_HIDDEN_DIM   = 8192;
constexpr int RMS_NUM_SMS      = 148;
constexpr int RMS_MAX_ROWS_PER_INST = 12;
constexpr int RMS_NUM_THREADS  = 256;

__host__ __device__ inline int rms_n_inst(int B) {
    if (B <= RMS_NUM_SMS) return B;
    int k = (B + RMS_MAX_ROWS_PER_INST * RMS_NUM_SMS - 1)
          / (RMS_MAX_ROWS_PER_INST * RMS_NUM_SMS);
    return RMS_NUM_SMS * k;
}

struct rms_config {
    static constexpr int HIDDEN_DIM  = RMS_HIDDEN_DIM;
    static constexpr int NUM_THREADS = RMS_NUM_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;  // bump when the kernel needs smem
};

struct rms_globals {
    using x_gl = kittens::gl<kittens::bf16, -1, -1, -1, -1>;
    using w_gl = kittens::gl<kittens::bf16, -1, -1, -1, -1>;
    using e_gl = kittens::gl<float,         -1, -1, -1, -1>;
    using o_gl = kittens::gl<kittens::bf16, -1, -1, -1, -1>;

    x_gl x;
    w_gl weight;
    e_gl eps;
    o_gl out;
};

// TODO(rms): implement the rmsnorm body.
//
// Layout: blockIdx.x in [0, n_inst), block handles rows [row_start, row_end)
//   row_start = (long long)blockIdx.x * B / n_inst
//   row_end   = (long long)(blockIdx.x + 1) * B / n_inst
// where B = g.x.rows() (the dynamic batch dim).
//
// For each row r in [row_start, row_end):
//   var = mean_c( x[r, c]^2 )  in fp32
//   inv = rsqrt(var + eps[0])
//   out[r, c] = (x[r, c] * inv) * weight[c]   (back to bf16)
//
// Suggested approach: 256 threads × 32 elements/thread covers the 8192-wide row
// with 16-byte vector loads; intra-block reduction via warp-shuffle then a
// single-warp shared-memory reduction.
__global__ void rms_kernel(const __grid_constant__ rms_globals g) {
    // placeholder body
}

inline void rms_dispatch(at::Tensor x, at::Tensor weight, at::Tensor eps, at::Tensor out) {
    CHECK_INPUT(x); CHECK_INPUT(weight); CHECK_INPUT(eps); CHECK_INPUT(out);
    TORCH_CHECK(x.dim() == 2, "x must be [B, hidden_dim]");
    TORCH_CHECK(out.sizes() == x.sizes(), "out must match x shape");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1),
                "weight must be [hidden_dim]");
    TORCH_CHECK(eps.dim() == 1 && eps.size(0) == 1, "eps must be [1] fp32");
    TORCH_CHECK(x.scalar_type()      == at::ScalarType::BFloat16, "x must be bf16");
    TORCH_CHECK(weight.scalar_type() == at::ScalarType::BFloat16, "weight must be bf16");
    TORCH_CHECK(eps.scalar_type()    == at::ScalarType::Float,    "eps must be fp32");
    TORCH_CHECK(out.scalar_type()    == at::ScalarType::BFloat16, "out must be bf16");
    TORCH_CHECK(x.size(1) == rms_config::HIDDEN_DIM,
                "hidden_dim mismatch; rebuild after updating RMS_HIDDEN_DIM");

    const int B = static_cast<int>(x.size(0));
    const int n_inst = rms_n_inst(B);

    rms_globals g{
        kittens::py::tensor_to_gl<rms_globals::x_gl>(x),
        kittens::py::tensor_to_gl<rms_globals::w_gl>(weight),
        kittens::py::tensor_to_gl<rms_globals::e_gl>(eps),
        kittens::py::tensor_to_gl<rms_globals::o_gl>(out),
    };

    if constexpr (rms_config::DYNAMIC_SHARED_MEMORY > 0) {
        cudaFuncSetAttribute(rms_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             rms_config::DYNAMIC_SHARED_MEMORY);
    }
    rms_kernel<<<n_inst, rms_config::NUM_THREADS, rms_config::DYNAMIC_SHARED_MEMORY>>>(g);
}

}  // namespace manual_kernels
