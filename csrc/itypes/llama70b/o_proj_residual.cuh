#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "itypes/llama70b/matmul_pipeline.cuh"

namespace megakittens {
namespace llama70b {

template <typename Config, typename Globals,
          int M, int N, int K,
          int Mb, int Nb, int Kb, int EPI_PIPE_DEPTH,
          int SRC_HIDDEN, int SRC_ATTN_OUT, int SRC_O_WEIGHTS, int DST_HIDDEN>
struct OProjResidual {
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

    struct pipeline_specifics {};

    using pipeline = matmul_pipeline<Config, Globals, M, N, K,
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
