#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "utils.cuh"
#include "itypes/llama70b/utils.cuh"

namespace megakittens {
namespace llama70b {

template <typename Config, typename Globals, int N,
          typename parsed_instruction, typename pipeline_specifics,
          int SRC_X, int SRC_WEIGHT, int SCALAR_EPS, int DST_Y>
struct rms_pipeline {
    static constexpr int WEIGHTS_PAGE = 0;
    static constexpr int ELEMS_PER_WARP = N / Config::NUM_CONSUMER_WARPS;
    static constexpr int RMS_SCRATCH_OFFSET = sizeof(kittens::sv_bf<N>);
    using row_vec = kittens::sv_bf<N>;
    using sv_slice_t = kittens::sv_bf<ELEMS_PER_WARP>;

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

    __device__ static inline row_vec &row_at(state_t<Config> &s, int row_idx) {
        int page_idx = 1 + row_idx / 2;
        int pos_in_page = row_idx % 2;
        int pid = s.lid_to_pid(page_idx);
        return *reinterpret_cast<row_vec*>(
            static_cast<uint8_t*>(s.pages[pid].ptr(pos_in_page * sizeof(row_vec))));
    }

    __device__ static inline void loader_loop(const Globals &g, state_t<Config> &s) {
        parsed_instruction inst{s};
        int lane = kittens::laneid();
        int num_used_pages = 1 + (inst.num_rows + 1) / 2;

        if (lane == 0) {
            all_input_barrier_wait<Config>(g, s.instruction());
            int weight_pid = s.lid_to_pid(WEIGHTS_PAGE);
            s.page_wait(weight_pid);
            row_vec &weight_smem = *reinterpret_cast<row_vec*>(s.pages[weight_pid].ptr());
            auto &w_gl = g.template gls<SRC_WEIGHT>();
            kittens::tma::expect_bytes(weights_arrived(s), sizeof(row_vec));
            kittens::tma::load_async(weight_smem, w_gl, {0, 0, 0, 0}, weights_arrived(s));

            auto &x_gl = g.template gls<SRC_X>();
            for (int i = 0; i < inst.num_rows; i++) {
                int page_idx = 1 + i / 2;
                int pos_in_page = i % 2;
                int row_pid = s.lid_to_pid(page_idx);
                if (pos_in_page == 0) s.page_wait(row_pid);
                row_vec &row_smem = row_at(s, i);
                auto &sem = activations_arrived(s, i);
                kittens::tma::expect_bytes(sem, sizeof(row_vec));
                kittens::tma::load_async(row_smem, x_gl, {0, 0, inst.row_start + i, 0}, sem);
            }
        } else if (lane >= num_used_pages && lane < Config::NUM_PAGES) {
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
        row_vec &weight_smem_full = *reinterpret_cast<row_vec*>(s.pages[weight_pid].ptr());
        sv_slice_t &weight_slice =
            reinterpret_cast<sv_slice_t *>(&weight_smem_full)[kittens::warpid()];
        float *rms_scratch = static_cast<float *>(s.pages[weight_pid].ptr(RMS_SCRATCH_OFFSET));
        float eps = g.template gls<SCALAR_EPS>().raw_ptr[0];

        for (int i = 0; i < inst.num_rows; i++) {
            kittens::wait(activations_arrived(s, i), 0);

            row_vec &row_smem = row_at(s, i);
            sv_slice_t &row_slice = reinterpret_cast<sv_slice_t *>(&row_smem)[kittens::warpid()];

            kittens::rv_fl<ELEMS_PER_WARP> act_vec;
            kittens::warp::load(act_vec, row_slice);
            float *row_scratch = rms_scratch + i * Config::NUM_CONSUMER_WARPS;
            act_vec = rms_norm<Config, N>(act_vec, weight_slice, eps, row_scratch);
            kittens::warp::store(row_slice, act_vec);
            kittens::warp::sync();
            kittens::warp::arrive(outputs_arrived(s, i));
        }
    }

    __device__ static inline void storer_loop(const Globals &g, state_t<Config> &s) {
        parsed_instruction inst{s};
        if (kittens::warp::elect_leader()) {
            all_reuse_barrier_wait<Config>(g, s.instruction());
            for (int i = 0; i < inst.num_rows; i++) {
                kittens::wait(outputs_arrived(s, i), 0);
                row_vec &row_smem = row_at(s, i);
                pipeline_specifics::store(s, g, inst, i, row_smem);
                kittens::tma::store_async_read_wait();
                int pos_in_page = i % 2;
                if (pos_in_page == 1 || i == inst.num_rows - 1) {
                    int page_idx = 1 + i / 2;
                    s.page_finish(s.lid_to_pid(page_idx));
                }
            }
            kittens::tma::store_async_wait();
            s.page_finish(s.lid_to_pid(WEIGHTS_PAGE));
            // TODO multi-gpu??: barrier scope needs to be `sys` for cross-GPU all-gather
            all_barrier_arrive<Config>(g, s.instruction());
        }
    }
};

}
}
