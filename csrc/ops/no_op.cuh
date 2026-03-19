
#pragma once

#include "kittens.cuh"

namespace megakittens {

template <typename Config, typename Globals>
struct NoOp {
    static constexpr int opcode = 0;

    struct controller {
        static __device__ int
        release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query) {
            return query;
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            return 0;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            if (kittens::laneid() < Config::NUM_PAGES) { // release all pages ASAP!
                auto pid = s.pid(kittens::laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }

            kittens::warp::arrive(s.instruction_fetch_ready, Config::NUM_CONSUMER_WARPS);
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {
            s.wait_tensor_ready();
            if (kittens::laneid() == 0)
                arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {}
    };

    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {}
    };
};

} // namespace megakittens
