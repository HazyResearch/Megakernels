#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include "matmul_pipeline.cuh"

namespace manual_kernels {

using namespace kittens;

template <typename C>
struct up_matmul_globals {
    using a_tile    = typename matmul_pipeline<C>::a_tile_t;
    using b_tile    = typename matmul_pipeline<C>::b_tile_t;
    using d_tile    = typename matmul_pipeline<C>::d_tile_t;
    using gate_tile = st_bf<C::Mb / 2, C::Nb / 2>;

    using a_gl    = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl    = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl    = gl<bf16, 1, 1, -1, -1, d_tile>;
    using gate_gl = gl<bf16, 1, 1, -1, -1, gate_tile>;

    a_gl    x;
    b_gl    up_w;
    gate_gl gate;
    d_gl    out;
};

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void up_matmul_kernel(const __grid_constant__ up_matmul_globals<C> g) {
    using P           = matmul_pipeline<C>;
    using a_tile_t    = typename P::a_tile_t;
    using b_tile_t    = typename P::b_tile_t;
    using d_tile_t    = typename P::d_tile_t;
    using d_tt_t      = typename P::d_tt_t;
    using gate_tile_t = typename up_matmul_globals<C>::gate_tile;
    using d_reg_t     = rt_fl<C::ROWS_PER_CONSUMER / 4, C::COLS_PER_CHUNK>;

    static_assert(C::EPI_PIPE_DEPTH % 2 == 0,
                  "gate is loaded as 2 halves so EPI_PIPE_DEPTH must be even");
    constexpr int CHUNKS_PER_HALF = C::EPI_PIPE_DEPTH / 2;


    const int cta_rank   = cluster_ctarank();
    const int num_iters  = g.x.cols() / C::Kb;
    const int N          = g.up_w.rows();
    const int cblks      = N / C::Nb;
    const int cluster_id = blockIdx.x / C::CLUSTER_SIZE;
    const int m          = cluster_id / cblks;
    const int n          = cluster_id % cblks;

    extern __shared__ int __shm_up[];
    tma_swizzle_allocator al((int*)&__shm_up[0]);

    auto &a_smem = al.allocate<a_tile_t, C::LOAD_PIPE_DEPTH, C::NUM_CONSUMERS>();
    auto &b_smem = al.allocate<b_tile_t, C::LOAD_PIPE_DEPTH>();
    // gate_smem aliases a_smem (128 KB == 128 KB). d_smem aliases the first 32 KB of b_smem.
    // Safe because gate is loaded only after the producer drains inputs_finished for all
    // matmul stages, and d_smem / b_smem are used only after wait(outputs_arrived).
    auto &gate_smem = *reinterpret_cast<
        gate_tile_t (*)[C::NUM_CONSUMERS][2]>(&a_smem[0][0]);
    auto &d_smem = *reinterpret_cast<
        d_tile_t (*)[C::NUM_CONSUMERS][C::NUM_D_TILES]>(&b_smem[0]);

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};

    __shared__ semaphore inputs_arrived [C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived[C::NUM_CONSUMERS];
    __shared__ semaphore gate_arrived   [C::NUM_CONSUMERS];
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
            init_semaphore(gate_arrived[i],    0, 1);
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
                             g.x, g.up_w);
            P::producer_drain(inputs_finished, bitfield, input_ring);
            #pragma unroll
            for (int cid = 0; cid < C::NUM_CONSUMERS; cid++) {
                tma::expect_bytes(gate_arrived[cid], 2 * sizeof(gate_tile_t));
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    tma::load_async(
                        gate_smem[cid][half], g.gate,
                        {0, 0, (2 * m + cta_rank) * C::NUM_CONSUMERS + cid, 2 * n + half},
                        gate_arrived[cid]);
                }
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
        wait(gate_arrived[cid], 0);

        #pragma unroll
        for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
            const int slot         = i % C::NUM_D_TILES;
            const int half         = i / CHUNKS_PER_HALF;
            const int col_chunk    = i % CHUNKS_PER_HALF;
            const int global_chunk = C::EPI_PIPE_DEPTH * n + i;

            d_reg_t d_reg;
            warpgroup::load_async(
                d_reg,
                d_tt.template subtile<tt<float, C::ROWS_PER_CONSUMER, C::COLS_PER_CHUNK>>(
                    0, C::COLS_PER_CHUNK * i));
            tensor_load_wait();

            d_reg_t gate_reg;
            warpgroup::load(
                gate_reg,
                gate_smem[cid][half].template subtile<C::ROWS_PER_CONSUMER, C::COLS_PER_CHUNK>(
                    {0, col_chunk}));

            warp::mul(d_reg, d_reg, gate_reg);

            warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
            warpgroup::sync(cid + 1);
            d_tile_t &out_tile = d_smem[cid][slot];
            warpgroup::store(out_tile, d_reg);
            warpgroup::sync(cid + 1);

            const int row_tile = (2 * m + cta_rank) * C::NUM_CONSUMERS + cid;
            warpgroup::tma::store_async(g.out, out_tile, {0, 0, row_tile, global_chunk});
        }

        warpgroup::tma::store_async_wait();
    }
}

inline void up_matmul_dispatch(
        at::Tensor x,
        at::Tensor up_w,
        at::Tensor gate,
        at::Tensor out) {
    using C = matmul_config</*Nb=*/256, /*LOAD_PIPE_DEPTH=*/4>;

    CHECK_INPUT(x); CHECK_INPUT(up_w); CHECK_INPUT(gate); CHECK_INPUT(out);
    TORCH_CHECK(x.dim() == 2, "x must be [M, K]");
    TORCH_CHECK(up_w.dim() == 3 && up_w.size(0) == 1, "up_w must be [1, N, K]");
    TORCH_CHECK(gate.dim() == 2, "gate must be [M, N]");
    TORCH_CHECK(out.dim() == 2, "out must be [M, N]");
    TORCH_CHECK(up_w.size(2) == x.size(1), "up_w K must match x K");
    TORCH_CHECK(gate.size(0) == x.size(0) && gate.size(1) == up_w.size(1),
                "gate must be [M, N]");
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == up_w.size(1),
                "out must be [M, N]");
    TORCH_CHECK(x.scalar_type()    == at::ScalarType::BFloat16);
    TORCH_CHECK(up_w.scalar_type() == at::ScalarType::BFloat16);
    TORCH_CHECK(gate.scalar_type() == at::ScalarType::BFloat16);
    TORCH_CHECK(out.scalar_type()  == at::ScalarType::BFloat16);

    const int M = static_cast<int>(x.size(0));
    const int N = static_cast<int>(up_w.size(1));
    const int K = static_cast<int>(x.size(1));
    TORCH_CHECK(M % C::M_INST == 0, "M must be divisible by M_INST (=512)");
    TORCH_CHECK(N % C::Nb     == 0, "N must be divisible by Nb (=256)");
    TORCH_CHECK(K % C::Kb     == 0, "K must be divisible by Kb (=64)");

    up_matmul_globals<C> g{
        kittens::py::tensor_to_gl<typename up_matmul_globals<C>::a_gl   >(x),
        kittens::py::tensor_to_gl<typename up_matmul_globals<C>::b_gl   >(up_w),
        kittens::py::tensor_to_gl<typename up_matmul_globals<C>::gate_gl>(gate),
        kittens::py::tensor_to_gl<typename up_matmul_globals<C>::d_gl   >(out),
    };

    constexpr int dyn_smem = MAX_SHARED_MEMORY - 1024;
    CUDACHECK(cudaFuncSetAttribute(
        up_matmul_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem));

    const int rblks = M / C::M_INST;
    const int cblks = N / C::Nb;
    const dim3 grid(rblks * cblks * C::CLUSTER_SIZE);
    const dim3 block(C::NUM_THREADS);
    up_matmul_kernel<C><<<grid, block, dyn_smem>>>(g);
}

}  // namespace manual_kernels
