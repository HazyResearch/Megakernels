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
    static_assert(Config::CLUSTER_SIZE == 2, "matmul_pipeline requires CLUSTER_SIZE == 2");

    static constexpr int NUM_CONSUMERS   = 1;
    static constexpr int LOAD_PIPE_DEPTH = 6;
    static constexpr int NUM_D_TILES     = 2;
    static constexpr int NUM_USED_PAGES  = 7;

    using a_st_t = kittens::st_bf<Mb / 2, Kb>;
    using b_st_t = kittens::st_bf<Nb / 2, Kb>;
    using d_st_t = kittens::st_bf<Mb / 2, Nb / EPI_PIPE_DEPTH>;
    using d_tt_t = kittens::tt<float, Mb / 2, Nb>;

    static constexpr int A_LIDS[LOAD_PIPE_DEPTH / 2] = {0, 2, 4};
    static constexpr int B_LIDS[LOAD_PIPE_DEPTH / 2] = {1, 3, 5};
    static constexpr int D_LID = 6;

    __device__ static inline kittens::semaphore &inputs_arrived (state_t<Config> &s, int stage) {
        return s.semaphores()[stage];
    }
    __device__ static inline kittens::semaphore &inputs_finished(state_t<Config> &s, int stage) {
        return s.semaphores()[LOAD_PIPE_DEPTH + stage];
    }
    __device__ static inline kittens::semaphore &outputs_arrived(state_t<Config> &s) {
        return s.semaphores()[2 * LOAD_PIPE_DEPTH];
    }

    __device__ static inline a_st_t &a_st(state_t<Config> &s, int stage) {
        return s.pages[s.lid_to_pid(A_LIDS[stage / 2])].template as<a_st_t>((stage % 2) * sizeof(a_st_t));
    }
    __device__ static inline b_st_t &b_st(state_t<Config> &s, int stage) {
        return s.pages[s.lid_to_pid(B_LIDS[stage / 2])].template as<b_st_t>((stage % 2) * sizeof(b_st_t));
    }
    __device__ static inline d_st_t &d_st(state_t<Config> &s, int slot) {
        return s.pages[s.lid_to_pid(D_LID)].template as<d_st_t>(slot * sizeof(d_st_t));
    }

    __device__ static inline int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
        static_assert(Config::NUM_PAGES == NUM_USED_PAGES && LOAD_PIPE_DEPTH == 6);
        const int num_iters = g.template gls<SRC_A>().cols() / Kb;
        switch ((num_iters % LOAD_PIPE_DEPTH) / 2) {
            case 0: { constexpr int order[] = {0, 1, 2, 3, 4, 5, 6}; return order[query]; }
            case 1: { constexpr int order[] = {2, 3, 4, 5, 0, 1, 6}; return order[query]; }
            case 2: { constexpr int order[] = {4, 5, 0, 1, 2, 3, 6}; return order[query]; }
        }
        return 0;
    }

    __device__ static inline int init_semaphores(const Globals &g, state_t<Config> &s) {
        const int lane_id = kittens::laneid();
        if (lane_id < LOAD_PIPE_DEPTH) {
            kittens::init_semaphore(inputs_arrived(s, lane_id), 1);
            kittens::init_semaphore(inputs_finished(s, lane_id), NUM_CONSUMERS);
        } else if (lane_id == LOAD_PIPE_DEPTH) {
            kittens::init_semaphore(outputs_arrived(s), 1);
        }
        return 2 * LOAD_PIPE_DEPTH + 1;
    }

    __device__ static inline void loader_loop(const Globals &g, state_t<Config> &s) {}
    __device__ static inline void launcher_run(const Globals &g, state_t<Config> &s) {}
    __device__ static inline void consumer_loop(const Globals &g, state_t<Config> &s) {}
    __device__ static inline void storer_loop(const Globals &g, state_t<Config> &s) {}
};

}
}
