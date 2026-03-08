
#pragma once

#include "util.cuh"

namespace megakernel {

template <typename config> struct NoOp {
    static constexpr int opcode = 0;

    struct controller {
        template <typename globals>
        static __device__ int
        release_lid(const globals &g,
                    typename config::instruction_t &instruction, int &query) {
            return query;
        }
        template <typename globals>
        static __device__ int init_semaphores(const globals &g,
                                              state<config> &s) {
            return 0;
        }
    };
    struct loader {
        template <typename globals>
        static __device__ void run(const globals &g, state<config> &s) {
            if (kittens::laneid() < config::NUM_PAGES) { // Release all pages, ASAP.
                auto pid = s.pid(kittens::laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, config::NUM_CONSUMER_WARPS);
            }

            kittens::warp::arrive(s.instruction_fetch_ready, config::NUM_CONSUMER_WARPS);
        }
    };
    struct launcher { // launches mma's
        // launcher does nothing here, since this doesn't use tensor cores.
        template <typename globals>
        static __device__ void run(const globals &g, state<config> &s) {
#ifdef KITTENS_BLACKWELL
            s.wait_tensor_ready();
            if (kittens::laneid() == 0)
                arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
#endif
        }
    };
    struct consumer {
        template <typename globals>
        static __device__ void run(const globals &g, state<config> &s) {}
    };
    struct storer {
        // Uses 4 full pages for outputs.
        template <typename globals>
        static __device__ void run(const globals &g, state<config> &s) {}
    };
};

} // namespace megakernel