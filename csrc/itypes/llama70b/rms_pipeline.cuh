#pragma once

#include "kittens.cuh"
#include "schema.cuh"

namespace megakittens {
namespace llama70b {

template <typename Config, typename Globals, int N,
          typename parsed_instruction, typename pipeline_specifics,
          int SRC0, int SRC1, int DST>
struct rms_pipeline {

    __device__ static inline int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
    }

    __device__ static inline int init_semaphores(const Globals &g, state_t<Config> &s) {
    }

    __device__ static inline void loader_loop(const Globals &g, state_t<Config> &s) {
    }

    __device__ static inline void launcher_run(const Globals &g, state_t<Config> &s) {
    }

    __device__ static inline void consumer_loop(const Globals &g, state_t<Config> &s) {
    }

    __device__ static inline void storer_loop(const Globals &g, state_t<Config> &s) {
    }
};

}
}
