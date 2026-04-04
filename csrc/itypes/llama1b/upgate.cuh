#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "utils.cuh"
#include "itypes/llama1b/matvec_pipeline.cuh"

namespace megakittens {

template <typename Config, typename Globals, int N, int SRC_ACT, int SRC_NORM, int SRC_UP, int SRC_GATE, int DST>
struct RmsUpgateSilu {

    struct parsed_instruction {
        int layer_idx, start_block_idx, end_block_idx, iters;
        __device__ inline parsed_instruction(const instruction_t &instruction) {
            layer_idx       = instruction.indices[0];
            start_block_idx = instruction.indices[1];
            end_block_idx   = instruction.indices[2];
            iters           = 2 * (end_block_idx - start_block_idx);
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
            int block_idx = inst.start_block_idx + iter / 2;
            if (iter % 2 == 0) {
                kittens::tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                    weight_chunk, g.template gls<SRC_UP>(),
                    {inst.layer_idx, block_idx, col_idx}, sem);
            } else {
                kittens::tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                    weight_chunk, g.template gls<SRC_GATE>(),
                    {inst.layer_idx, block_idx, col_idx}, sem);
            }
        }

        __device__ static inline void
        store(state_t<Config> &s, const Globals &g, parsed_instruction &inst,
              int output_idx, int output_stage) {
            if (output_idx % 2 == 0) {
                return;
            }

            int true_output_idx = output_idx / 2;

            // NOTE: hardcoding to 3 output stages for now
            int prev_output_idx = (output_idx - 1);
            int prev_output_stage = prev_output_idx % 3;

            int block_idx = inst.start_block_idx + true_output_idx;

            uint8_t *output_scratch = pipeline::get_output_start(s, output_stage);
            uint8_t *prev_output_scratch = pipeline::get_output_start(s, prev_output_stage);

            kittens::sv_bf<16> &out_smem =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch);

            kittens::rv_fl<16> up_out, gate_out, gate_scratch;

            // TODO we can do better here and reduce up before gate is ready.
            llama1b::matvec_reduce<Config, pipeline::SCRATCH_BYTES_PER_WARP>(
                prev_output_scratch, up_out);
            llama1b::matvec_reduce<Config, pipeline::SCRATCH_BYTES_PER_WARP>(
                output_scratch, gate_out);

            kittens::warp::mul(gate_scratch, gate_out, -1.f);
            kittens::warp::exp(gate_scratch, gate_scratch);
            kittens::warp::add(gate_scratch, gate_scratch, 1.f);
            kittens::warp::div(gate_out, gate_out, gate_scratch);
            kittens::warp::mul(gate_out, up_out, gate_out);
            kittens::warp::sync();
            kittens::warp::store(out_smem, gate_out);
            kittens::warp::sync();

            if (kittens::warp::elect_leader()) {
                kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                    g.template gls<DST>(), out_smem, {0, block_idx});
                kittens::tma::store_async_wait();
                // atomic add here
            }

            kittens::warp::sync();
        }
    };

    using pipeline = llama1b::rms_matvec_pipeline<
        Config, Globals, N, parsed_instruction, pipeline_specifics, SRC_ACT, SRC_NORM>;
    static_assert(pipeline::OUTPUT_PIPELINE_STAGES == 3);

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
            parsed_instruction inst{s};
            pipeline::loader_loop(s, g, inst.layer_idx);
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
            pipeline::template storer_loop<2>(s, g);
            // atomic add here
            if (kittens::warp::elect_leader()) {
                __threadfence();
                all_barrier_arrive<Config>(g, s.instruction());
            }
        }
    };
};

} // namespace megakittens
