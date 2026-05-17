#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include "matmul_pipeline.cuh"

namespace manual_kernels {

using namespace kittens;

// lm_head: logits = hidden @ w[0].T. Pure matmul + TMA store. N is the vocab
// size (128256 for Llama-3.3-70B), so at B=1024 the grid is large enough
// (2 × 501 × 2 = 2004 blocks ≈ 13.5 waves on B200) that no special scheduling
// is needed.
template <typename C>
struct lm_head_globals {
    using a_tile = typename matmul_pipeline<C>::a_tile_t;
    using b_tile = typename matmul_pipeline<C>::b_tile_t;
    using d_tile = typename matmul_pipeline<C>::d_tile_t;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

    a_gl hidden;
    b_gl w;
    d_gl logits;
};

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void lm_head_kernel(const __grid_constant__ lm_head_globals<C> g) {
    using P        = matmul_pipeline<C>;
    using a_tile_t = typename P::a_tile_t;
    using b_tile_t = typename P::b_tile_t;
    using d_tile_t = typename P::d_tile_t;
    using d_tt_t   = typename P::d_tt_t;
    using d_reg_t  = rt_bf<C::ROWS_PER_CONSUMER / 4, C::COLS_PER_CHUNK>;

    const int cta_rank   = cluster_ctarank();
    const int num_iters  = g.hidden.cols() / C::Kb;
    const int N          = g.w.rows();
    const int cblks      = N / C::Nb;
    const int cluster_id = blockIdx.x / C::CLUSTER_SIZE;
    const int m          = cluster_id / cblks;
    const int n          = cluster_id % cblks;

    extern __shared__ int __shm_lm[];
    tma_swizzle_allocator al((int*)&__shm_lm[0]);

    auto &a_smem = al.allocate<a_tile_t, C::LOAD_PIPE_DEPTH, C::NUM_CONSUMERS>();
    auto &b_smem = al.allocate<b_tile_t, C::LOAD_PIPE_DEPTH>();
    auto &d_smem = *reinterpret_cast<
        d_tile_t (*)[C::NUM_CONSUMERS][C::NUM_D_TILES]>(&a_smem[0][0]);

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};

    __shared__ semaphore inputs_arrived [C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived[C::NUM_CONSUMERS];
    uint32_t bitfield = 0xFFFF0000;

    if (threadIdx.x == 32) {
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i],  0, C::NUM_CONSUMERS);
            init_semaphore(inputs_finished[i], 0, C::NUM_CONSUMERS);
        }
        #pragma unroll
        for (int i = 0; i < C::NUM_CONSUMERS; i++) {
            init_semaphore(outputs_arrived[i], 0, 1);
        }
    }
    everyone::tma::cluster::arrive_aligned();

    if (warpgroup::groupid() == C::NUM_CONSUMERS) {
        warpgroup::decrease_registers<56>();

        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            int input_ring = 0;
            P::producer_load(a_smem, b_smem, inputs_arrived, inputs_finished,
                             bitfield, input_ring, num_iters, cta_rank, m, n,
                             g.hidden, g.w);
        } else if (cta_rank == 0 && warpgroup::warpid() < C::NUM_CONSUMERS && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            d_tt_t d_tt = tm_alloc.template allocate<d_tt_t>(warpgroup::warpid() * C::Nb);
            P::launcher_mma(a_smem, b_smem, inputs_arrived, inputs_finished,
                            outputs_arrived[warpgroup::warpid()],
                            d_tt, bitfield, num_iters, warpgroup::warpid());
        }
    } else {
        const int cid = warpgroup::groupid();

        warpgroup::increase_registers<224>();
        everyone::tma::cluster::wait_aligned();

        d_tt_t d_tt = tm_alloc.template allocate<d_tt_t>(cid * C::Nb);
        wait(outputs_arrived[cid], 0);

        #pragma unroll
        for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
            const int slot         = i % C::NUM_D_TILES;
            const int global_chunk = C::EPI_PIPE_DEPTH * n + i;

            d_reg_t d_reg;
            warpgroup::load_async(
                d_reg,
                d_tt.template subtile<tt<float, C::ROWS_PER_CONSUMER, C::COLS_PER_CHUNK>>(
                    0, C::COLS_PER_CHUNK * i));
            tensor_load_wait();

            warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
            warpgroup::sync(cid + 1);
            d_tile_t &out_tile = d_smem[cid][slot];
            warpgroup::store(out_tile, d_reg);
            warpgroup::sync(cid + 1);

            const int row_tile = (2 * m + cta_rank) * C::NUM_CONSUMERS + cid;
            warpgroup::tma::store_async(g.logits, out_tile, {0, 0, row_tile, global_chunk});
        }

        warpgroup::tma::store_async_wait();
    }
}

inline void lm_head_dispatch(
        at::Tensor hidden,
        at::Tensor w,
        at::Tensor logits) {
    using C = matmul_config</*Nb=*/256, /*LOAD_PIPE_DEPTH=*/4>;

    CHECK_INPUT(hidden); CHECK_INPUT(w); CHECK_INPUT(logits);
    TORCH_CHECK(hidden.dim() == 2, "hidden must be [M, K]");
    TORCH_CHECK(w.dim() == 3 && w.size(0) == 1, "w must be [1, N, K]");
    TORCH_CHECK(logits.dim() == 2, "logits must be [M, N]");
    TORCH_CHECK(w.size(2) == hidden.size(1), "w K must match hidden K");
    TORCH_CHECK(logits.size(0) == hidden.size(0), "logits M must match hidden M");
    TORCH_CHECK(logits.size(1) == w.size(1), "logits N must match w N");
    TORCH_CHECK(hidden.scalar_type() == at::ScalarType::BFloat16);
    TORCH_CHECK(w.scalar_type()      == at::ScalarType::BFloat16);
    TORCH_CHECK(logits.scalar_type() == at::ScalarType::BFloat16);

    const int M = static_cast<int>(hidden.size(0));
    const int N = static_cast<int>(w.size(1));
    const int K = static_cast<int>(hidden.size(1));
    TORCH_CHECK(M % C::M_INST == 0, "M must be divisible by M_INST (=512)");
    TORCH_CHECK(N % C::Nb     == 0, "N must be divisible by Nb (=256)");
    TORCH_CHECK(K % C::Kb     == 0, "K must be divisible by Kb (=64)");

    lm_head_globals<C> g{
        kittens::py::tensor_to_gl<typename lm_head_globals<C>::a_gl>(hidden),
        kittens::py::tensor_to_gl<typename lm_head_globals<C>::b_gl>(w),
        kittens::py::tensor_to_gl<typename lm_head_globals<C>::d_gl>(logits),
    };

    constexpr int dyn_smem = MAX_SHARED_MEMORY - 1024;
    CUDACHECK(cudaFuncSetAttribute(
        lm_head_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem));

    const int rblks = M / C::M_INST;
    const int cblks = N / C::Nb;
    const dim3 grid(rblks * cblks * C::CLUSTER_SIZE);
    const dim3 block(C::NUM_THREADS);
    lm_head_kernel<C><<<grid, block, dyn_smem, at::cuda::getCurrentCUDAStream()>>>(g);
}

}  // namespace manual_kernels
