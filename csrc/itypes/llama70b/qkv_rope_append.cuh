#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "itypes/llama70b/matmul_pipeline.cuh"

namespace megakittens {
namespace llama70b {

template <typename Config, typename Globals, int N, int HEAD_DIM, int NUM_KV_HEADS,
          int SRC_X, int SRC_QKV_W, int SRC_ROPE_COS, int SRC_ROPE_SIN, int SRC_POS_IDS,
          int SRC_APPEND_IDS, int DST_Q, int DST_K_CACHE = -1, int DST_V_CACHE = -1>
struct QkvRopeAppend {
    static_assert(HEAD_DIM == 128, "Head dim must be 128.");

    static constexpr int HEADS_PER_INST = 2;
    static constexpr int KV_COL_START = (N / HEAD_DIM) / HEADS_PER_INST;

    static constexpr int Mb = 128;
    static constexpr int Nb = 256;
    static constexpr int Kb = 64;
    static constexpr int EPI_PIPE_DEPTH = 2;

    using rope_sv_t = kittens::sv_fl<HEAD_DIM>;
    using rope_rv_t = kittens::rv_fl<HEAD_DIM>;
    using head_sv_t = kittens::sv_fl<HEAD_DIM>;
    using head_rv_t = kittens::rv_fl<HEAD_DIM>;

    struct parsed_instruction {
        int layer_idx, m, n;
        __device__ inline parsed_instruction(const instruction_t &instruction) {
            layer_idx = instruction.indices[0];
            m = instruction.indices[1];
            n = instruction.indices[2];
        }
        __device__ inline parsed_instruction(state_t<Config> &s)
            : parsed_instruction(s.instruction()) {}

        __device__ inline bool is_kv() const { return n >= KV_COL_START; }
    };

    __device__ static inline void apply_rope_inplace(head_rv_t &x,
                                                     const head_rv_t &cos_v,
                                                     const head_rv_t &sin_v) {}

    struct pipeline_specifics {
        template <typename Pipeline>
        __device__ static inline void consumer_loop(const Globals &g, state_t<Config> &s) {}
    };

    using pipeline = matmul_pipeline<Config, Globals,
                                     /*M=*/0, /*N=*/N, /*K=*/0,
                                     Mb, Nb, Kb, EPI_PIPE_DEPTH,
                                     parsed_instruction, pipeline_specifics,
                                     SRC_X, SRC_QKV_W>;

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

} // namespace llama70b
} // namespace megakittens
