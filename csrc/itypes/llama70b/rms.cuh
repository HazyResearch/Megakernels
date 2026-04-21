#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "itypes/llama70b/rms_pipeline.cuh"

namespace megakittens {
namespace llama70b {

template <typename Config, typename Globals, int N, int SRC0, int SRC1, int DST>
struct RMS {
    struct parsed_instruction {
        int row_start, num_rows;
        __device__ inline parsed_instruction(const instruction_t &instruction) {
            row_start = instruction.indices[2];
            num_rows  = instruction.indices[3];
        }
        __device__ inline parsed_instruction(state_t<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

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
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {}
    };
};

}
}
