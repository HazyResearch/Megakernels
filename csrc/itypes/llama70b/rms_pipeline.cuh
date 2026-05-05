#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "utils.cuh"
#include "itypes/llama70b/utils.cuh"

namespace megakittens {
namespace llama70b {

template <typename Config, typename Globals, int N,
          typename parsed_instruction,
          int SRC_X, int SRC_WEIGHT, int SCALAR_EPS, int DST_Y>
struct rms_pipeline {
    static constexpr int WEIGHTS_PAGE = 0;
    static constexpr int ELEMS_PER_WARP = N / Config::NUM_CONSUMER_WARPS;
    static constexpr int RMS_SCRATCH_OFFSET = sizeof(kittens::sv_bf<N>);
    using row_vec = kittens::sv_bf<N>;
    using sv_slice_t = kittens::sv_bf<ELEMS_PER_WARP>;
    using consumer_group = kittens::group<Config::NUM_CONSUMER_WARPS>;

    __device__ static inline kittens::semaphore &weights_arrived(state_t<Config> &s) { return s.semaphores()[0]; }

    __device__ static inline int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
        // Must be num_rows-independent so cluster-paired CTAs agree on pid_order for the next instruction's MMA
        constexpr int order[Config::NUM_PAGES] = {1, 2, 3, 4, 5, 6, 0};
        return order[query];
    }

    __device__ static inline int init_semaphores(const Globals &g, state_t<Config> &s) {
        if (kittens::laneid() == 0) kittens::init_semaphore(weights_arrived(s), 1);
        return 1;
    }

    __device__ static inline void loader_loop(const Globals &g, state_t<Config> &s) {
        int lane = kittens::laneid();

        if (lane == 0) {
            parsed_instruction inst{s};
            all_input_barrier_wait<Config>(g, s.instruction());
            all_reuse_barrier_wait<Config>(g, s.instruction());

            int weight_pid = s.lid_to_pid(WEIGHTS_PAGE);
            s.page_wait(weight_pid);
            row_vec &weight_smem = *reinterpret_cast<row_vec*>(s.pages[weight_pid].ptr());
            auto &w_gl = g.template gls<SRC_WEIGHT>();
            kittens::tma::expect_bytes(weights_arrived(s), sizeof(row_vec));
            kittens::tma::load_async(weight_smem, w_gl, {0, 0, inst.layer_idx, 0}, weights_arrived(s));
        } else if (lane >= 1 && lane < Config::NUM_PAGES) {
            int pid = s.lid_to_pid(lane);
            s.page_wait(pid);
            s.page_finish(pid);
        }
    }

    __device__ static inline void launcher_run(const Globals &g, state_t<Config> &s) {
        s.tensor_wait();
        if (kittens::warp::elect_leader()) s.tensor_finish();
    }

    __device__ static inline void consumer_loop(const Globals &g, state_t<Config> &s) {
        parsed_instruction inst{s};
        kittens::wait(weights_arrived(s), 0);

        int weight_pid = s.lid_to_pid(WEIGHTS_PAGE);
        row_vec &weight_smem = *reinterpret_cast<row_vec*>(s.pages[weight_pid].ptr());
        sv_slice_t &weight_slice =
            reinterpret_cast<sv_slice_t *>(&weight_smem)[kittens::warpid()];
        float *rms_scratch = static_cast<float *>(s.pages[weight_pid].ptr(RMS_SCRATCH_OFFSET));
        float eps = g.template gls<SCALAR_EPS>().raw_ptr[0];

        auto &x_gl = g.template gls<SRC_X>();
        auto &y_gl = g.template gls<DST_Y>();

        for (int i = 0; i < inst.num_rows; i++) {
            kittens::rv_fl<ELEMS_PER_WARP> act_vec;
            consumer_group::load(act_vec, x_gl, {0, 0, inst.row_start + i, 0});

            float *row_scratch = rms_scratch + i * Config::NUM_CONSUMER_WARPS;
            act_vec = rms_norm<Config, N>(act_vec, weight_slice, eps, row_scratch);

            consumer_group::store(y_gl, act_vec, {0, 0, inst.row_start + i, 0});
        }

        consumer_group::sync(2);
        if (consumer_group::elect_leader()) {
            __threadfence();
            s.page_finish(s.lid_to_pid(WEIGHTS_PAGE));
            // TODO multi-gpu??: barrier scope needs to be `sys` for cross-GPU all-gather
            all_barrier_arrive<Config>(g, s.instruction());
        }
    }

    __device__ static inline void storer_loop(const Globals &g, state_t<Config> &s) {}
};

}
}
