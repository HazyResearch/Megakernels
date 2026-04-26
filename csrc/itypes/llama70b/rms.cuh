#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "itypes/llama70b/rms_pipeline.cuh"

namespace megakittens {
namespace llama70b {

template <typename Config, typename Globals, int N, int SRC_X, int SRC_WEIGHT, int SCALAR_EPS, int DST_Y>
struct RMS {
    struct parsed_instruction {
        int layer_idx, row_start, num_rows;
        __device__ inline parsed_instruction(const instruction_t &instruction) {
            layer_idx = instruction.indices[0];
            row_start = instruction.indices[2];
            num_rows  = instruction.indices[3];
        }
        __device__ inline parsed_instruction(state_t<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        __device__ static inline void store(state_t<Config> &s, const Globals &g,
                                            parsed_instruction &inst, int row_idx,
                                            kittens::sv_bf<N> &row_smem) {
            auto &y_gl = g.template gls<DST_Y>();
            kittens::tma::store_async(y_gl, row_smem, {0, 0, inst.row_start + row_idx, 0});
        }
    };

    using pipeline = rms_pipeline<Config, Globals, N, parsed_instruction, pipeline_specifics,
                                  SRC_X, SRC_WEIGHT, SCALAR_EPS, DST_Y>;

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            return pipeline::lid_release_order(g, s, query);
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            return pipeline::init_semaphores(g, s);
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::loader_loop(g, s);
        }
    };

    struct launcher {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::launcher_run(g, s);
        }
    };

    struct consumer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::consumer_loop(g, s);
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::storer_loop(g, s);
        }
    };
};

}
}
