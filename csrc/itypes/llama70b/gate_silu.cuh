#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "itypes/llama70b/matmul_pipeline.cuh"

namespace megakittens {
namespace llama70b {

template <typename Config, typename Globals,
          int M, int N, int K,
          int Mb, int Nb, int Kb, int EPI_PIPE_DEPTH,
          int SRC_X, int SRC_W, int DST_OUT>
struct GateSilu {
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

    struct pipeline_specifics {
        template <typename Pipeline>
        __device__ static inline void consumer_loop(const Globals &g, state_t<Config> &s) {
            parsed_instruction pi{s};
            const int cta_rank = kittens::cluster_ctarank();
            const int cid = kittens::warpgroup::groupid();
            using consumer_group = kittens::group<kittens::WARPGROUP_WARPS * Pipeline::NUM_CONSUMERS>;

            auto &out_gl = g.template gls<DST_OUT>();

            typename Pipeline::d_tt_t d_tt = s.tensor_alloc.template allocate<typename Pipeline::d_tt_t>(cid * Nb);
            kittens::wait(Pipeline::outputs_arrived(s), 0);

            if (consumer_group::elect_leader()) all_reuse_barrier_wait<Config>(g, s.instruction());
            consumer_group::sync(4);

            #pragma unroll
            for (int i = 0; i < EPI_PIPE_DEPTH; i++) {
                const int slot = i % Pipeline::NUM_D_TILES;

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

                kittens::warpgroup::tma::store_async_read_wait<Pipeline::NUM_D_TILES - 1>();
                kittens::warpgroup::sync(cid + 1);
                kittens::warpgroup::store(Pipeline::d_st(s, cid, slot), d_reg);
                kittens::warpgroup::sync(cid + 1);
                kittens::warpgroup::tma::store_async(
                    out_gl, Pipeline::d_st(s, cid, slot),
                    {0, 0, (2 * pi.m + cta_rank) * Pipeline::NUM_CONSUMERS + cid, EPI_PIPE_DEPTH * pi.n + i});
            }

            if (consumer_group::elect_leader()) s.tensor_finish();

            kittens::warpgroup::tma::store_async_wait();
            consumer_group::sync(4);
            if (consumer_group::elect_leader()) {
                s.page_finish(s.lid_to_pid(Pipeline::A_LIDS[0]));
                all_barrier_arrive<Config>(g, s.instruction());
            }
        }
    };

    using pipeline = matmul_pipeline<Config, Globals, M, N, K,
                                     Mb, Nb, Kb, EPI_PIPE_DEPTH,
                                     parsed_instruction, pipeline_specifics,
                                     SRC_X, SRC_W,
                                     kittens::cache_policy::EVICT_LAST,
                                     kittens::cache_policy::EVICT_FIRST>;

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            return pipeline::lid_release_order(g, s, query);
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            return pipeline::init_semaphores(g, s);
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::loader_loop(g, s);
        }
    };

    struct launcher {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::launcher_run(g, s);
        }
    };

    struct consumer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::consumer_loop(g, s);
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::storer_loop(g, s);
        }
    };
};

}
}
