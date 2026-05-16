#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

namespace manual_kernels {

using namespace kittens;

template <int _Nb, int _LOAD_PIPE_DEPTH>
struct gate_silu_config {
    static constexpr int Mb              = 256;
    static constexpr int Nb              = _Nb;
    static constexpr int Kb              = 64;
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

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;

    static_assert(COLS_PER_CHUNK == 32, "epilogue assumes 32-col chunks");
    static_assert(Nb % 32 == 0, "Nb must be divisible by 32 (2-CTA weight loader)");
    static_assert(Nb <= 256, "Nb <= 256");
    static_assert(Nb % COLS_PER_CHUNK == 0);
};

template <typename C>
struct gate_silu_globals {
    using a_tile = st_bf<C::Mb / 2, C::Kb>;
    using b_tile = st_bf<C::Nb / 2, C::Kb>;
    using d_tile = st_bf<C::Mb / 2, C::COLS_PER_CHUNK>;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

    a_gl x;
    b_gl gate_w;
    d_gl out;
};

struct silu_op {
    template <typename T> static __device__ inline T op(const T &x) {
        if constexpr (std::is_same_v<T, float>) {
            return __fdividef(x, 1.f + __expf(-x));
        } else if constexpr (std::is_same_v<T, float2>) {
            return float2{__fdividef(x.x, 1.f + __expf(-x.x)),
                          __fdividef(x.y, 1.f + __expf(-x.y))};
        }
    }
};

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void gate_silu_kernel(const __grid_constant__ gate_silu_globals<C> g) {
    using G        = gate_silu_globals<C>;
    using a_tile_t = typename G::a_tile;
    using b_tile_t = typename G::b_tile;
    using d_tile_t = typename G::d_tile;
    using d_tt_t   = tt<float, C::Mb / 2, C::Nb>;
    using d_reg_t  = rt_fl<C::ROWS_PER_CONSUMER / 4, C::COLS_PER_CHUNK>;

    if (threadIdx.x == 0) {
        g.x.template      prefetch_tma<a_tile_t>();
        g.gate_w.template prefetch_tma<b_tile_t>();
        g.out.template    prefetch_tma<d_tile_t>();
    }

    const int cta_rank   = cluster_ctarank();
    const int num_iters  = g.x.cols() / C::Kb;
    const int N          = g.gate_w.rows();
    const int cblks      = N / C::Nb;
    const int cluster_id = blockIdx.x / C::CLUSTER_SIZE;
    const int m          = cluster_id / cblks;
    const int n          = cluster_id % cblks;

    extern __shared__ int __shm_gs[];
    tma_swizzle_allocator al((int*)&__shm_gs[0]);

    a_tile_t (&a_smem)[C::LOAD_PIPE_DEPTH][C::NUM_CONSUMERS] =
        al.allocate<a_tile_t, C::LOAD_PIPE_DEPTH, C::NUM_CONSUMERS>();
    b_tile_t (&b_smem)[C::LOAD_PIPE_DEPTH]                   =
        al.allocate<b_tile_t, C::LOAD_PIPE_DEPTH>();
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
            for (int idx = 0; idx < num_iters; idx++) {
                wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                #pragma unroll
                for (int i = 0; i < C::NUM_CONSUMERS; i++) {
                    tma::cluster::load_async(
                        a_smem[input_ring][i], g.x,
                        {(2 * m + cta_rank) * C::NUM_CONSUMERS + i, idx},
                        inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                }
                tma::cluster::load_async(
                    b_smem[input_ring], g.gate_w,
                    {0, 0, 2 * n + cta_rank, idx},
                    inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                update_phasebit<1>(bitfield, input_ring);
                input_ring = ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
            }
        } else if (cta_rank == 0 && warpgroup::warpid() < C::NUM_CONSUMERS && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            d_tt_t d_tt = tm_alloc.template allocate<d_tt_t>(warpgroup::warpid() * C::Nb);

            int input_ring = 0;
            for (int idx = 0; idx < num_iters; idx++) {
                tma::expect_bytes(
                    inputs_arrived[input_ring],
                    (C::CLUSTER_SIZE * C::NUM_CONSUMERS * sizeof(a_tile_t)
                     + 2 * sizeof(b_tile_t)) / C::NUM_CONSUMERS);
                wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                if (idx == 0)
                    mm2_ABt (d_tt, a_smem[input_ring][warpgroup::warpid()],
                             b_smem[input_ring], inputs_finished[input_ring]);
                else
                    mma2_ABt(d_tt, a_smem[input_ring][warpgroup::warpid()],
                             b_smem[input_ring], inputs_finished[input_ring]);
                update_phasebit<0>(bitfield, input_ring);
                input_ring = ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
            }
            detail::tcgen05::commit<C::CLUSTER_SIZE>(outputs_arrived[warpgroup::warpid()]);
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

            warp::unary_map<silu_op>(d_reg, d_reg);

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

inline void gate_silu_dispatch(
        at::Tensor x,
        at::Tensor gate_w,
        at::Tensor out) {
    using C = gate_silu_config</*Nb=*/256, /*LOAD_PIPE_DEPTH=*/4>;

    CHECK_INPUT(x); CHECK_INPUT(gate_w); CHECK_INPUT(out);
    TORCH_CHECK(x.dim() == 2, "x must be [M, K]");
    TORCH_CHECK(gate_w.dim() == 3 && gate_w.size(0) == 1, "gate_w must be [1, N, K]");
    TORCH_CHECK(out.dim() == 2, "out must be [M, N]");
    TORCH_CHECK(gate_w.size(2) == x.size(1), "gate_w K must match x K");
    TORCH_CHECK(out.size(0) == x.size(0), "out M must match x M");
    TORCH_CHECK(out.size(1) == gate_w.size(1), "out N must match gate_w N");
    TORCH_CHECK(x.scalar_type()      == at::ScalarType::BFloat16);
    TORCH_CHECK(gate_w.scalar_type() == at::ScalarType::BFloat16);
    TORCH_CHECK(out.scalar_type()    == at::ScalarType::BFloat16);

    const int M = static_cast<int>(x.size(0));
    const int N = static_cast<int>(gate_w.size(1));
    const int K = static_cast<int>(x.size(1));
    TORCH_CHECK(M % C::M_INST == 0, "M must be divisible by M_INST (=512)");
    TORCH_CHECK(N % C::Nb     == 0, "N must be divisible by Nb (=256)");
    TORCH_CHECK(K % C::Kb     == 0, "K must be divisible by Kb (=64)");

    gate_silu_globals<C> g{
        kittens::py::tensor_to_gl<typename gate_silu_globals<C>::a_gl>(x),
        kittens::py::tensor_to_gl<typename gate_silu_globals<C>::b_gl>(gate_w),
        kittens::py::tensor_to_gl<typename gate_silu_globals<C>::d_gl>(out),
    };

    constexpr int dyn_smem = MAX_SHARED_MEMORY - 1024;
    CUDACHECK(cudaFuncSetAttribute(
        gate_silu_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem));

    const int rblks = M / C::M_INST;
    const int cblks = N / C::Nb;
    const dim3 grid(rblks * cblks * C::CLUSTER_SIZE);
    const dim3 block(C::NUM_THREADS);
    gate_silu_kernel<C><<<grid, block, dyn_smem>>>(g);
}

}  // namespace manual_kernels
