#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "itypes/llama70b/matmul_pipeline_half_tmem.cuh"

namespace megakittens {
namespace llama70b {

template <typename Config, typename Globals,
          int M, int N, int K,
          int Mb, int Nb, int Kb, int EPI_PIPE_DEPTH,
          int SRC_HIDDEN, int SRC_ATTN_OUT, int SRC_O_WEIGHTS, int DST_HIDDEN>
struct OProjResidualHalfTmem {
    static constexpr int COLS_PER_CHUNK = Nb / EPI_PIPE_DEPTH;
    static_assert(Nb > 0, "Nb must be positive.");
    static_assert(Nb <= 256, "OProjResidualHalfTmem70b valid Nb values are <= 256.");
    static_assert(Nb % 32 == 0, "OProjResidualHalfTmem70b requires Nb divisible by 32.");
    static_assert(Nb % EPI_PIPE_DEPTH == 0, "Nb must divide evenly into epilogue chunks.");
    static_assert(COLS_PER_CHUNK == 32, "OProjResidualHalfTmem70b epilogue assumes 32-column chunks.");

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
            const int wg_id = kittens::warpgroup::groupid();
            using consumer_group = kittens::group<Config::NUM_CONSUMER_WARPS>;
            constexpr int CHUNKS_PER_WG = (EPI_PIPE_DEPTH + 1) / 2;

            auto &hidden_gl = g.template gls<DST_HIDDEN>();

            typename Pipeline::d_tt_t d_tt = s.tensor_alloc.template allocate<typename Pipeline::d_tt_t>(0);
            kittens::wait(Pipeline::outputs_arrived(s), 0);

            kittens::rt_bf<Mb / 8, Nb / EPI_PIPE_DEPTH> d_reg[CHUNKS_PER_WG];
            #pragma unroll
            for (int j = 0; j < CHUNKS_PER_WG; j++) {
                const int chunk = 2 * j + wg_id;
                if (chunk < EPI_PIPE_DEPTH) {
                    kittens::warpgroup::load_async(
                        d_reg[j],
                        d_tt.template subtile<kittens::tt<float, Mb / 2, Nb / EPI_PIPE_DEPTH>>(0, (Nb / EPI_PIPE_DEPTH) * chunk));
                }
            }
            kittens::tensor_load_wait();
            if (consumer_group::elect_leader()) all_reuse_barrier_wait<Config>(g, s.instruction());
            consumer_group::sync(4);
            if (consumer_group::elect_leader()) s.tensor_finish();

            #pragma unroll
            for (int j = 0; j < CHUNKS_PER_WG; j++) {
                const int chunk = 2 * j + wg_id;
                const int slot  = j % Pipeline::NUM_D_TILES;
                if (chunk < EPI_PIPE_DEPTH) {
                    kittens::warpgroup::tma::store_async_read_wait<Pipeline::NUM_D_TILES - 1>();
                    kittens::warpgroup::sync(wg_id + 1);
                    kittens::warpgroup::store(Pipeline::d_st(s, wg_id, slot), d_reg[j]);
                    kittens::warpgroup::sync(wg_id + 1);
                    kittens::warpgroup::tma::store_add_async(
                        hidden_gl, Pipeline::d_st(s, wg_id, slot),
                        {0, 0, 2 * pi.m + cta_rank, EPI_PIPE_DEPTH * pi.n + chunk});
                }
            }

            kittens::warpgroup::tma::store_async_wait();
            consumer_group::sync(4);
            if (consumer_group::elect_leader()) {
                s.page_finish(s.lid_to_pid(Pipeline::D_LID));
                all_barrier_arrive<Config>(g, s.instruction());
            }
        }
    };

    using pipeline = matmul_pipeline_half_tmem<Config, Globals, M, N, K,
                                               Mb, Nb, Kb, EPI_PIPE_DEPTH,
                                               parsed_instruction, pipeline_specifics,
                                               SRC_ATTN_OUT, SRC_O_WEIGHTS>;

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
