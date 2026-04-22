#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "itypes/llama70b/utils.cuh"

namespace megakittens {
namespace llama70b {

template <typename Config, typename Globals,
          int M, int N, int K,
          int Mb, int Nb, int Kb, int EPI_PIPE_DEPTH,
          int SRC_X, int SRC_W, int SRC_GATE, int DST_OUT>
struct UpMatmul {
    static_assert(Config::CLUSTER_SIZE == 2, "UpMatmul requires CLUSTER_SIZE == 2");

    static constexpr int NUM_CONSUMERS   = 2;
    static constexpr int LOAD_PIPE_DEPTH = 4;
    static constexpr int NUM_D_TILES     = 2;
    static constexpr int NUM_USED_PAGES  = 6;

    using a_st_t = kittens::st_bf<Mb / 2, Kb>;
    using b_st_t = kittens::st_bf<Nb / 2, Kb>;
    using d_st_t = kittens::st_bf<Mb / 2, Nb / EPI_PIPE_DEPTH>;
    using d_tt_t = kittens::tt<float, Mb / 2, Nb>;

    static constexpr int A_LIDS[LOAD_PIPE_DEPTH]     = {0, 2, 3, 5};
    static constexpr int B_LIDS[LOAD_PIPE_DEPTH / 2] = {1, 4};

    __device__ static inline kittens::semaphore &inputs_arrived (state_t<Config> &s, int stage) {
        return s.semaphores()[stage];
    }
    __device__ static inline kittens::semaphore &inputs_finished(state_t<Config> &s, int stage) {
        return s.semaphores()[LOAD_PIPE_DEPTH + stage];
    }
    __device__ static inline kittens::semaphore &outputs_arrived(state_t<Config> &s) {
        return s.semaphores()[2 * LOAD_PIPE_DEPTH];
    }

    __device__ static inline a_st_t &a_st(state_t<Config> &s, int stage, int cid) {
        return s.pages[s.lid_to_pid(A_LIDS[stage])].template as<a_st_t>(cid * sizeof(a_st_t));
    }
    __device__ static inline b_st_t &b_st(state_t<Config> &s, int stage) {
        return s.pages[s.lid_to_pid(B_LIDS[stage / 2])].template as<b_st_t>((stage % 2) * sizeof(b_st_t));
    }
    __device__ static inline d_st_t &d_st(state_t<Config> &s, int cid, int slot) {
        return s.pages[s.lid_to_pid(A_LIDS[0])].template as<d_st_t>((cid * NUM_D_TILES + slot) * sizeof(d_st_t));
    }

    struct parsed_instruction {
        int layer_idx, m, n;
        __device__ inline parsed_instruction(const instruction_t &instruction) {
            layer_idx = instruction.indices[0];
            m = instruction.indices[1];
            n = instruction.indices[2];
        }
        __device__ inline parsed_instruction(state_t<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            static_assert(Config::NUM_PAGES == 7 && LOAD_PIPE_DEPTH == 4);
            const int num_iters = g.template gls<SRC_X>().cols() / Kb;
            switch (num_iters % LOAD_PIPE_DEPTH) {
                case 0: case 1: { constexpr int order[] = {6, 2, 1, 3, 5, 4, 0}; return order[query]; }
                case 2:         { constexpr int order[] = {6, 3, 5, 4, 2, 1, 0}; return order[query]; }
                case 3:         { constexpr int order[] = {6, 5, 4, 2, 1, 3, 0}; return order[query]; }
            }
            return 0;
        }

        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            const int lane_id = kittens::laneid();
            if (lane_id < LOAD_PIPE_DEPTH) {
                kittens::init_semaphore(inputs_arrived(s, lane_id), 1);
                kittens::init_semaphore(inputs_finished(s, lane_id), NUM_CONSUMERS);
            } else if (lane_id == LOAD_PIPE_DEPTH) {
                kittens::init_semaphore(outputs_arrived(s), 1);
            }
            return 2 * LOAD_PIPE_DEPTH + 1;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            parsed_instruction pi{s};
            const int cta_rank = kittens::cluster_ctarank();
            auto &a_gl = g.template gls<SRC_X>();
            auto &b_gl = g.template gls<SRC_W>();
            const int num_iters = a_gl.cols() / Kb;

            if (kittens::warp::elect_leader()) {
                all_input_barrier_wait<Config>(g, s.instruction());
                for (int i = 0; i < num_iters + LOAD_PIPE_DEPTH; i++) {
                    const int stage = i % LOAD_PIPE_DEPTH;
                    if (i < LOAD_PIPE_DEPTH) {
                        s.page_wait(s.lid_to_pid(A_LIDS[stage]));
                        if (stage % 2 == 0) s.page_wait(s.lid_to_pid(B_LIDS[stage / 2]));
                    } else {
                        kittens::wait(inputs_finished(s, stage),
                                      ((i + LOAD_PIPE_DEPTH) / LOAD_PIPE_DEPTH) & 0b1);
                    }
                    if (i < num_iters) {
                        #pragma unroll
                        for (int cid = 0; cid < NUM_CONSUMERS; cid++) {
                            kittens::tma::cluster::load_async(
                                a_st(s, stage, cid), a_gl,
                                {0, 0, (2 * pi.m + cta_rank) * NUM_CONSUMERS + cid, i},
                                inputs_arrived(s, stage), (uint16_t)(1 << cta_rank), 0);
                        }
                        kittens::tma::cluster::load_async(
                            b_st(s, stage), b_gl,
                            {0, pi.layer_idx, 2 * pi.n + cta_rank, i},
                            inputs_arrived(s, stage), (uint16_t)(1 << cta_rank), 0);
                    } else {
                        if (stage != 0) s.page_finish(s.lid_to_pid(A_LIDS[stage]));
                        if (stage % 2 == 1) s.page_finish(s.lid_to_pid(B_LIDS[(stage - 1) / 2]));
                    }
                }
            } else if (kittens::warp::elect_leader_from_active()) {
                #pragma unroll
                for (int i = NUM_USED_PAGES; i < Config::NUM_PAGES; i++) {
                    s.page_wait(s.lid_to_pid(i));
                    s.page_finish(s.lid_to_pid(i));
                }
            }
        }
    };

    struct launcher {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const int cta_rank = kittens::cluster_ctarank();
            auto &a_gl = g.template gls<SRC_X>();
            const int num_iters = a_gl.cols() / Kb;

            if (cta_rank == 0 && kittens::warp::elect_leader()) {
                d_tt_t d_tt[NUM_CONSUMERS];
                #pragma unroll
                for (int cid = 0; cid < NUM_CONSUMERS; cid++) {
                    d_tt[cid] = s.tensor_alloc.template allocate<d_tt_t>(cid * Nb);
                }
                s.tensor_wait();
                for (int i = 0; i < num_iters; i++) {
                    const int stage = i % LOAD_PIPE_DEPTH;
                    kittens::tma::expect_bytes(inputs_arrived(s, stage),
                                               2 * NUM_CONSUMERS * sizeof(a_st_t) + 2 * sizeof(b_st_t));
                    kittens::wait(inputs_arrived(s, stage), (i / LOAD_PIPE_DEPTH) & 0b1);
                    #pragma unroll
                    for (int cid = 0; cid < NUM_CONSUMERS; cid++) {
                        if (i == 0) kittens::mm2_ABt (d_tt[cid], a_st(s, stage, cid), b_st(s, stage), inputs_finished(s, stage));
                        else        kittens::mma2_ABt(d_tt[cid], a_st(s, stage, cid), b_st(s, stage), inputs_finished(s, stage));
                    }
                }
                kittens::tensor_commit<Config::CLUSTER_SIZE>(outputs_arrived(s));
            }
        }
    };

    struct consumer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            parsed_instruction pi{s};
            const int cta_rank = kittens::cluster_ctarank();
            const int cid = kittens::warpgroup::groupid();
            using consumer_group = kittens::group<kittens::WARPGROUP_WARPS * NUM_CONSUMERS>;

            auto &out_gl = g.template gls<DST_OUT>();

            d_tt_t d_tt = s.tensor_alloc.template allocate<d_tt_t>(cid * Nb);
            kittens::wait(outputs_arrived(s), 0);

            if (consumer_group::elect_leader()) all_reuse_barrier_wait<Config>(g, s.instruction());
            consumer_group::sync(4);

            #pragma unroll
            for (int i = 0; i < EPI_PIPE_DEPTH; i++) {
                const int slot = i % NUM_D_TILES;

                kittens::rt_fl<Mb / 8, Nb / EPI_PIPE_DEPTH> d_reg;
                kittens::warpgroup::load_async(
                    d_reg,
                    d_tt.template subtile<kittens::tt<float, Mb / 2, Nb / EPI_PIPE_DEPTH>>(0, (Nb / EPI_PIPE_DEPTH) * i));
                kittens::tensor_load_wait();

                kittens::rt_fl<Mb / 8, Nb / EPI_PIPE_DEPTH> silu_buf;
                kittens::warp::mul(silu_buf, d_reg, -1.f);
                kittens::warp::exp(silu_buf, silu_buf);
                kittens::warp::add(silu_buf, silu_buf, 1.f);
                kittens::warp::div(d_reg, d_reg, silu_buf);

                kittens::warpgroup::tma::store_async_read_wait<NUM_D_TILES - 1>();
                kittens::warpgroup::sync(cid + 1);
                kittens::warpgroup::store(d_st(s, cid, slot), d_reg);
                kittens::warpgroup::sync(cid + 1);
                kittens::warpgroup::tma::store_async(
                    out_gl, d_st(s, cid, slot),
                    {0, 0, (2 * pi.m + cta_rank) * NUM_CONSUMERS + cid, EPI_PIPE_DEPTH * pi.n + i});
            }

            if (consumer_group::elect_leader()) s.tensor_finish();

            kittens::warpgroup::tma::store_async_wait();
            consumer_group::sync(4);
            if (consumer_group::elect_leader()) {
                s.page_finish(s.lid_to_pid(A_LIDS[0]));
                all_barrier_arrive<Config>(g, s.instruction());
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {}
    };
};

}
}
