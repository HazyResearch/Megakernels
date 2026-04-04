#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "utils.cuh"
#include "itypes/llama1b/matvec_pipeline.cuh"

namespace megakittens {

template <typename Config, typename Globals, int N, int SRC0, int SRC1, int SRC2, int DST>
struct RmsLmHead {

    struct parsed_instruction {
        int start_block_idx, end_block_idx, iters;
        __device__ inline parsed_instruction(const instruction_t &instruction) {
            start_block_idx = instruction.indices[0];
            end_block_idx   = instruction.indices[1];
            iters           = end_block_idx - start_block_idx;
        }
        __device__ inline parsed_instruction(state_t<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        __device__ static inline void gmem_wait(const Globals &g, state_t<Config> &s) {
            all_input_barrier_wait<Config>(g, s.instruction());
        }

        __device__ static inline void
        load_iter(state_t<Config> &s, const Globals &g, parsed_instruction &inst,
                  int iter, int col_idx,
                  kittens::st_bf<16, 512> &weight_chunk,
                  kittens::semaphore &sem) {
            int block_idx = inst.start_block_idx + iter;
            kittens::tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                                     weight_chunk, g.template gls<SRC2>(),
                                     {block_idx, col_idx}, sem);
        }

        __device__ static inline void
        store(state_t<Config> &s, const Globals &g, parsed_instruction &inst,
              int output_idx, int output_stage) {
            int block_idx = inst.start_block_idx + output_idx;

            uint8_t *output_scratch = pipeline::get_output_start(s, output_stage);

            kittens::rv_fl<16> logits_rv;
            llama1b::matvec_reduce<Config, pipeline::SCRATCH_BYTES_PER_WARP>(
                output_scratch, logits_rv);

            // Reuse output scratch as TMA staging (reduce already read from it)
            kittens::sv_bf<16> &logits_smem =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch);

            kittens::warp::sync();
            kittens::warp::store(logits_smem, logits_rv);
            kittens::warp::sync();

            if (kittens::warp::elect_leader()) {
                kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(g.template gls<DST>(), logits_smem, {0, block_idx});
                kittens::tma::store_async_read_wait();
            }
            kittens::warp::sync();
        }
    };

    using pipeline = llama1b::rms_matvec_pipeline<
        Config, Globals, N, parsed_instruction, pipeline_specifics, SRC0, SRC1>;

    struct controller {
        __device__ __forceinline__ static int
        lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            return pipeline::lid_release_order(g, s, query);
        }
        __device__ __forceinline__ static int
        init_semaphores(const Globals &g, state_t<Config> &s) {
            return pipeline::init_semaphores(g, s);
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::loader_loop(s, g);
        }
    };

    struct launcher {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::launcher_loop(s, g);
        }
    };

    struct consumer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::consumer_loop(s, g);
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::storer_loop(s, g);
        }
    };
};

} // namespace megakittens
