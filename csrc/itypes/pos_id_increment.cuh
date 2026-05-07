#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "utils.cuh"

namespace megakittens {

template <typename Config, typename Globals, int SRC, int DST>
struct PosIdIncrement {
    static_assert(SRC == DST, "PosIdIncrement: SRC must alias DST");

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            return query;
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            return 0;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            if (kittens::laneid() < Config::NUM_PAGES) {
                int pid = s.lid_to_pid(kittens::laneid());
                s.page_wait(pid);
                s.page_finish(pid);
            }
        }
    };

    struct launcher {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            s.tensor_wait();
            if (kittens::warp::elect_leader()) s.tensor_finish();
        }
    };

    struct consumer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {}
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            if (kittens::warp::elect_leader()) {
                all_input_barrier_wait<Config>(g, s.instruction());
                all_reuse_barrier_wait<Config>(g, s.instruction());
                g.template gls<SRC>().raw_ptr[0] += 1;
                __threadfence();
                all_barrier_arrive<Config>(g, s.instruction());
            }
        }
    };
};

} // namespace megakittens
