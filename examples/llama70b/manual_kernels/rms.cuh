#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

namespace manual_kernels {

using namespace kittens;

constexpr int RMS_HIDDEN_DIM        = 8192;
constexpr int RMS_NUM_SMS           = 148;
constexpr int RMS_NUM_WARPS         = 8;
constexpr int RMS_NUM_THREADS       = RMS_NUM_WARPS * WARP_THREADS;
constexpr int RMS_ELEMS_PER_WARP    = RMS_HIDDEN_DIM / RMS_NUM_WARPS;  // 1024
constexpr int RMS_MAX_ROWS_PER_INST = 12;

__host__ __device__ inline int rms_n_inst(int B) {
    if (B <= RMS_NUM_SMS) return B;
    int k = (B + RMS_MAX_ROWS_PER_INST * RMS_NUM_SMS - 1)
          / (RMS_MAX_ROWS_PER_INST * RMS_NUM_SMS);
    return RMS_NUM_SMS * k;
}

struct rms_config {
    static constexpr int HIDDEN_DIM     = RMS_HIDDEN_DIM;
    static constexpr int NUM_WARPS      = RMS_NUM_WARPS;
    static constexpr int NUM_THREADS    = RMS_NUM_THREADS;
    static constexpr int ELEMS_PER_WARP = RMS_ELEMS_PER_WARP;
};

struct rms_globals {
    using x_gl = gl<bf16,  -1, -1, -1, -1>;
    using w_gl = gl<bf16,  -1, -1, -1, -1>;
    using e_gl = gl<float, -1, -1, -1, -1>;
    using o_gl = gl<bf16,  -1, -1, -1, -1>;

    x_gl x;
    w_gl weight;
    e_gl eps;
    o_gl out;
};

__launch_bounds__(rms_config::NUM_THREADS, 1)
__global__ void rms_kernel(const __grid_constant__ rms_globals g) {
    constexpr int N  = rms_config::HIDDEN_DIM;
    constexpr int NW = rms_config::NUM_WARPS;
    constexpr int E  = rms_config::ELEMS_PER_WARP;

    using row_vec        = sv_bf<N>;
    using slice_vec      = sv_bf<E>;
    using consumer_group = group<NW>;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    row_vec &weight_smem = al.allocate<row_vec>();
    __shared__ float scratch[NW];

    const int B      = g.x.rows();
    const int n_inst = gridDim.x;
    const int row_lo = (int)(((long long)blockIdx.x       * B) / n_inst);
    const int row_hi = (int)(((long long)(blockIdx.x + 1) * B) / n_inst);
    const int num_rows = row_hi - row_lo;

    consumer_group::load_async(weight_smem, g.weight, {0, 0, 0, 0});
    load_async_wait();
    consumer_group::sync(1);

    slice_vec &weight_slice = reinterpret_cast<slice_vec *>(&weight_smem)[warpid()];
    const float eps_val = g.eps.raw_ptr[0];

    for (int i = 0; i < num_rows; i++) {
        rv_fl<E> act_vec;
        consumer_group::load(act_vec, g.x, {0, 0, row_lo + i, 0});

        rv_fl<E> sq;
        warp::copy(sq, act_vec);
        warp::mul(sq, sq, sq);
        float partial = warp::sum(sq);

        if (warp::elect_leader()) scratch[warpid()] = partial;
        consumer_group::sync(1);

        float full = 0.f;
        #pragma unroll
        for (int w = 0; w < NW; w++) full += scratch[w];
        const float rms_scale = rsqrtf(full / float(N) + eps_val);

        rv_fl<E> w_vec;
        warp::load(w_vec, weight_slice);
        warp::mul(w_vec, w_vec, rms_scale);
        warp::mul(act_vec, act_vec, w_vec);

        consumer_group::store(g.out, act_vec, {0, 0, row_lo + i, 0});
        consumer_group::sync(1);
    }
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

    constexpr int dyn_smem = MAX_SHARED_MEMORY - 1024;
    CUDACHECK(cudaFuncSetAttribute(
        rms_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem));
    rms_kernel<<<n_inst, rms_config::NUM_THREADS, dyn_smem, at::cuda::getCurrentCUDAStream()>>>(g);
}

}  // namespace manual_kernels
