#pragma once

#include "kittens.cuh"

namespace manual_kernels {

using namespace kittens;

template <int _Nb, int _LOAD_PIPE_DEPTH>
struct matmul_config {
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
    static constexpr int LOAD_PIPE_DEPTH   = _LOAD_PIPE_DEPTH;

    static_assert(COLS_PER_CHUNK == 32, "epilogue assumes 32-col chunks");
    static_assert(Nb % 32 == 0, "Nb must be divisible by 32 (2-CTA weight loader)");
    static_assert(Nb <= 256, "Nb <= 256");
    static_assert(Nb % COLS_PER_CHUNK == 0);
};

template <typename C>
struct matmul_pipeline {
    using a_tile_t = st_bf<C::Mb / 2, C::Kb>;
    using b_tile_t = st_bf<C::Nb / 2, C::Kb>;
    using d_tile_t = st_bf<C::Mb / 2, C::COLS_PER_CHUNK>;
    using d_tt_t   = tt<float, C::Mb / 2, C::Nb>;

    using a_gl_t = gl<bf16, 1, 1, -1, -1, a_tile_t>;
    using b_gl_t = gl<bf16, 1, 1, -1, -1, b_tile_t>;

    using a_smem_t = a_tile_t[C::LOAD_PIPE_DEPTH][C::NUM_CONSUMERS];
    using b_smem_t = b_tile_t[C::LOAD_PIPE_DEPTH];

    // Per-consumer share of the cluster's A/B bytes (inputs_arrived expects NUM_CONSUMERS arrivals).
    static constexpr int A_BYTES_PER_ARRIVAL =
        (C::CLUSTER_SIZE * C::NUM_CONSUMERS * sizeof(a_tile_t)
         + 2 * sizeof(b_tile_t)) / C::NUM_CONSUMERS;

    // Stream A and B over K. Call from one elected thread in the producer warpgroup.
    __device__ static inline void producer_load(
            a_smem_t &a_smem, b_smem_t &b_smem,
            semaphore *inputs_arrived, semaphore *inputs_finished,
            uint32_t &bitfield, int &input_ring,
            int num_iters, int cta_rank, int m, int n,
            const a_gl_t &a_gl, const b_gl_t &b_gl) {
        for (int idx = 0; idx < num_iters; idx++) {
            wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
            #pragma unroll
            for (int i = 0; i < C::NUM_CONSUMERS; i++) {
                tma::cluster::load_async(
                    a_smem[input_ring][i], a_gl,
                    {(2 * m + cta_rank) * C::NUM_CONSUMERS + i, idx},
                    inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
            }
            tma::cluster::load_async(
                b_smem[input_ring], b_gl,
                {0, 0, 2 * n + cta_rank, idx},
                inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
            update_phasebit<1>(bitfield, input_ring);
            input_ring = ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
        }
    }

    // Drain the final LOAD_PIPE_DEPTH MMA stages. Use after producer_load when
    // the producer reuses a/b smem in its epilogue (e.g. aliased gate/residual loads).
    __device__ static inline void producer_drain(
            semaphore *inputs_finished, uint32_t &bitfield, int &input_ring) {
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; i++) {
            wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
            update_phasebit<1>(bitfield, input_ring);
            input_ring = ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
        }
    }

    __device__ static inline void launcher_mma(
            a_smem_t &a_smem, b_smem_t &b_smem,
            semaphore *inputs_arrived, semaphore *inputs_finished,
            semaphore &my_outputs_arrived,
            d_tt_t d_tt, uint32_t &bitfield,
            int num_iters, int wg_warpid) {
        int input_ring = 0;
        for (int idx = 0; idx < num_iters; idx++) {
            tma::expect_bytes(inputs_arrived[input_ring], A_BYTES_PER_ARRIVAL);
            wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
            if (idx == 0)
                mm2_ABt (d_tt, a_smem[input_ring][wg_warpid],
                         b_smem[input_ring], inputs_finished[input_ring]);
            else
                mma2_ABt(d_tt, a_smem[input_ring][wg_warpid],
                         b_smem[input_ring], inputs_finished[input_ring]);
            update_phasebit<0>(bitfield, input_ring);
            input_ring = ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
        }
        detail::tcgen05::commit<C::CLUSTER_SIZE>(my_outputs_arrived);
    }
};

}  // namespace manual_kernels
