#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "itypes/llama70b/utils.cuh"

namespace megakittens {
namespace llama70b {

template <typename Config, typename Globals,
          int M, int N, int K,
          int Mb, int Nb, int Kb, int EPI_PIPE_DEPTH,
          typename parsed_instruction, typename pipeline_specifics,
          int SRC_A, int SRC_B>
struct matmul_pipeline {
    __device__ static inline int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
        return 0;
    }

    __device__ static inline int init_semaphores(const Globals &g, state_t<Config> &s) {
        return 0;
    }

    __device__ static inline void loader_loop(const Globals &g, state_t<Config> &s) {}
    __device__ static inline void launcher_run(const Globals &g, state_t<Config> &s) {}
    __device__ static inline void consumer_loop(const Globals &g, state_t<Config> &s) {}
    __device__ static inline void storer_loop(const Globals &g, state_t<Config> &s) {}
};

}
}
