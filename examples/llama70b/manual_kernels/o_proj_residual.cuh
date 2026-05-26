#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include "matmul_pipeline.cuh"

namespace manual_kernels {

using namespace kittens;

// Local Kb-tunable config for the o_proj/down_proj kernel only. Mirrors the
// fields matmul_pipeline<C> consumes off the shared `matmul_config`, but
// auto-picks LOAD_PIPE_DEPTH from the smem budget so that sweeping Kb keeps
// the deepest pipeline that still fits.
template <int _Kb = 64>
struct o_proj_config {
    static constexpr int Mb              = 256;
    static constexpr int Nb              = 256;
    static constexpr int Kb              = _Kb;
    static constexpr int COLS_PER_CHUNK  = 32;
    static constexpr int EPI_PIPE_DEPTH  = Nb / COLS_PER_CHUNK;
    static constexpr int NUM_D_TILES     = 2;

    static constexpr int CLUSTER_SIZE      = 2;
    static constexpr int NUM_CONSUMERS     = 2;
    static constexpr int NUM_PRODUCERS     = 1;
    static constexpr int NUM_WARPGROUPS    = NUM_CONSUMERS + NUM_PRODUCERS;
    static constexpr int NUM_WARPS         = NUM_WARPGROUPS * 4;
    static constexpr int NUM_THREADS       = NUM_WARPS * WARP_THREADS;
    static constexpr int M_INST            = NUM_CONSUMERS * Mb;
    static constexpr int ROWS_PER_CONSUMER = Mb / 2;

    // Per-stage smem = a_smem + b_smem
    //                = LPD × NUM_CONSUMERS × (Mb/2 × Kb × 2 bytes)
    //                + LPD × (Nb/2 × Kb × 2 bytes)
    //                = LPD × Kb × (2 × Mb + Nb).
    // Budget matches the dispatch's `MAX_SHARED_MEMORY - 1024`.
    // Hard ceiling at 16: the phasebit encoding in the kernel packs phases
    // into a 32-bit `bitfield` (16 bits per phase), so ring depth > 16 would
    // silently corrupt phase tracking.
    static constexpr int BYTES_PER_STAGE = Kb * (2 * Mb + Nb);
    static constexpr int SMEM_BUDGET     = MAX_SHARED_MEMORY - 1024;
    static constexpr int LPD_FROM_SMEM   = SMEM_BUDGET / BYTES_PER_STAGE;
    static constexpr int LOAD_PIPE_DEPTH = LPD_FROM_SMEM < 16 ? LPD_FROM_SMEM : 16;

    static_assert(Kb >= 16 && Kb % 16 == 0,
                  "Kb must be a positive multiple of 16 (tcgen05.mma bf16 K granularity)");
    static_assert(LOAD_PIPE_DEPTH >= 2,
                  "smem budget too small for >=2 pipeline stages");
};

// o_proj_residual: hidden += attn_out @ o_w[0].T.
// Atomically adds the matmul output into the residual using tma::store_add_async,
// so the same tensor (hidden) is both an input and the destination. The matmul
// pipeline itself is identical to gate_silu's; the epilogue just stores-with-add.
template <typename C>
struct o_proj_residual_globals {
    using a_tile = typename matmul_pipeline<C>::a_tile_t;
    using b_tile = typename matmul_pipeline<C>::b_tile_t;
    using d_tile = typename matmul_pipeline<C>::d_tile_t;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

    a_gl attn_out;
    b_gl o_w;
    d_gl hidden;
};

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void o_proj_residual_kernel(const __grid_constant__ o_proj_residual_globals<C> g) {
    using P        = matmul_pipeline<C>;
    using a_tile_t = typename P::a_tile_t;
    using b_tile_t = typename P::b_tile_t;
    using d_tile_t = typename P::d_tile_t;
    using d_tt_t   = typename P::d_tt_t;
    using d_reg_t  = rt_bf<C::ROWS_PER_CONSUMER / 4, C::COLS_PER_CHUNK>;

    const int cta_rank   = cluster_ctarank();
    const int num_iters  = g.attn_out.cols() / C::Kb;
    const int N          = g.o_w.rows();
    const int cblks      = N / C::Nb;
    const int cluster_id = blockIdx.x / C::CLUSTER_SIZE;
    const int m          = cluster_id / cblks;
    const int n          = cluster_id % cblks;

    extern __shared__ int __shm_op[];
    tma_swizzle_allocator al((int*)&__shm_op[0]);

    auto &a_smem = al.allocate<a_tile_t, C::LOAD_PIPE_DEPTH, C::NUM_CONSUMERS>();
    auto &b_smem = al.allocate<b_tile_t, C::LOAD_PIPE_DEPTH>();
    // d_smem aliases a_smem[0]; safe because outputs_arrived only flips after
    // all matmul iters finish reading every a/b stage.
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
            // Inlined producer_load with per-operand L2 cache policy:
            //   A (attn_out): EVICT_LAST -- reused 32x across cblks within this kernel.
            //   B (o_w / down_w): EVICT_FIRST -- only 2x reuse at ~32-cluster distance;
            //   marking evict_first avoids polluting L2 for the next kernel.
            for (int idx = 0; idx < num_iters; idx++) {
                wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                #pragma unroll
                for (int i = 0; i < C::NUM_CONSUMERS; i++) {
                    tma::cluster::load_async<dim::ROW, cache_policy::EVICT_LAST>(
                        a_smem[input_ring][i], g.attn_out,
                        {(2 * m + cta_rank) * C::NUM_CONSUMERS + i, idx},
                        inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                }
                tma::cluster::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    b_smem[input_ring], g.o_w,
                    {0, 0, 2 * n + cta_rank, idx},
                    inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                update_phasebit<1>(bitfield, input_ring);
                input_ring = ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
            }
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
            // EVICT_LAST: `hidden` is read by rms immediately after this kernel.
            warpgroup::tma::store_add_async<dim::ROW, cache_policy::EVICT_LAST>(
                g.hidden, out_tile, {0, 0, row_tile, global_chunk});
        }

        warpgroup::tma::store_async_wait();
    }
}

inline void o_proj_residual_dispatch(
        at::Tensor attn_out,
        at::Tensor o_w,
        at::Tensor hidden) {
    using C = o_proj_config</*Kb=*/64>;

    CHECK_INPUT(attn_out); CHECK_INPUT(o_w); CHECK_INPUT(hidden);
    TORCH_CHECK(attn_out.dim() == 2, "attn_out must be [M, K]");
    TORCH_CHECK(o_w.dim() == 3 && o_w.size(0) == 1, "o_w must be [1, N, K]");
    TORCH_CHECK(hidden.dim() == 2, "hidden must be [M, N]");
    TORCH_CHECK(o_w.size(2) == attn_out.size(1), "o_w K must match attn_out K");
    TORCH_CHECK(hidden.size(0) == attn_out.size(0) && hidden.size(1) == o_w.size(1),
                "hidden must be [M, N]");
    TORCH_CHECK(attn_out.scalar_type() == at::ScalarType::BFloat16);
    TORCH_CHECK(o_w.scalar_type()      == at::ScalarType::BFloat16);
    TORCH_CHECK(hidden.scalar_type()   == at::ScalarType::BFloat16);

    const int M = static_cast<int>(attn_out.size(0));
    const int N = static_cast<int>(o_w.size(1));
    const int K = static_cast<int>(attn_out.size(1));
    TORCH_CHECK(M % C::M_INST == 0, "M must be divisible by M_INST (=512)");
    TORCH_CHECK(N % C::Nb     == 0, "N must be divisible by Nb (=256)");
    TORCH_CHECK(K % C::Kb     == 0, "K must be divisible by Kb (=64)");

    o_proj_residual_globals<C> g{
        kittens::py::tensor_to_gl<typename o_proj_residual_globals<C>::a_gl>(attn_out),
        kittens::py::tensor_to_gl<typename o_proj_residual_globals<C>::b_gl>(o_w),
        kittens::py::tensor_to_gl<typename o_proj_residual_globals<C>::d_gl>(hidden),
    };

    constexpr int dyn_smem = MAX_SHARED_MEMORY - 1024;
    CUDACHECK(cudaFuncSetAttribute(
        o_proj_residual_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem));

    const int rblks = M / C::M_INST;
    const int cblks = N / C::Nb;
    const dim3 grid(rblks * cblks * C::CLUSTER_SIZE);
    const dim3 block(C::NUM_THREADS);
    o_proj_residual_kernel<C><<<grid, block, dyn_smem, at::cuda::getCurrentCUDAStream()>>>(g);
}

}  // namespace manual_kernels
