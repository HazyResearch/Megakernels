#pragma once

#include "kittens.cuh"
#include "schema.cuh"

namespace megakittens {
namespace llama70b {

template <typename Config, typename Globals, int N,
          typename parsed_instruction, typename pipeline_specifics,
          int SRC0, int SRC1, int DST>
struct rms_pipeline {
    static constexpr int WEIGHTS_PAGE = 0;
    using row_vec = kittens::sv_bf<N>;

    __device__ static inline kittens::semaphore &weights_arrived(state_t<Config> &s)        { return s.semaphores()[0]; }
    __device__ static inline kittens::semaphore &activations_arrived(state_t<Config> &s, int i) { return s.semaphores()[2 * i + 1]; }
    __device__ static inline kittens::semaphore &outputs_arrived(state_t<Config> &s, int i)     { return s.semaphores()[2 * i + 2]; }

    __device__ static inline int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
        parsed_instruction inst{s};
        int num_row_pages = (inst.num_rows + 1) / 2;
        int num_used = 1 + num_row_pages;
        int num_unused = Config::NUM_PAGES - num_used;
        if (query < num_unused)            return num_used + query;
        if (query < Config::NUM_PAGES - 1) return query - num_unused + 1;
        return 0;
    }

    __device__ static inline int init_semaphores(const Globals &g, state_t<Config> &s) {
        parsed_instruction inst{s};
        if (kittens::laneid() == 0) kittens::init_semaphore(weights_arrived(s), 1);
        if (kittens::laneid() < inst.num_rows) {
            kittens::init_semaphore(activations_arrived(s, kittens::laneid()), 1);
            kittens::init_semaphore(outputs_arrived(s, kittens::laneid()), Config::NUM_CONSUMER_WARPS);
        }
        return 2 * inst.num_rows + 1;
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
