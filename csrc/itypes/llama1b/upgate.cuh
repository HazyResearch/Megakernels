#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "utils.cuh"
#include "itypes/llama1b/matvec_pipeline.cuh"

namespace megakittens {

template <typename Config, typename Globals, int N, int SRC_ACT, int SRC_NORM, int SRC_UP, int SRC_GATE, int DST>
struct RmsUpgateSilu {

    struct parsed_instruction {
        int layer_idx, sm_idx, sm_count, total_blocks, barrier_base, iters;
        __device__ inline parsed_instruction(const instruction_t &instruction) {
            layer_idx    = instruction.indices[0];
            sm_idx       = instruction.indices[1];
            sm_count     = instruction.indices[2];
            total_blocks = instruction.indices[3];
            barrier_base = instruction.indices[4];
            iters        = 2 * ((total_blocks - sm_idx + sm_count - 1) / sm_count);
        }
        __device__ inline parsed_instruction(state_t<Config> &s)
            : parsed_instruction(s.instruction()) {}
        __device__ inline int block_at(int i) const {
            return sm_idx + i * sm_count;
        }
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
            int block_idx = inst.block_at(iter / 2);
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
            // unused — storer inlines the loop to cache up_out across iterations
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
            constexpr int OPS = pipeline::OUTPUT_PIPELINE_STAGES;
            parsed_instruction inst{s};
            kittens::rv_fl<16> up_out;
            int output_stage = 0;

            for (int i = 0; i < inst.iters; i++) {
                kittens::wait(pipeline::outputs_arrived(s, output_stage),
                    (i % (2 * OPS)) >= OPS);

                if (i % 2 == 0) {
                    // up iteration: reduce into register now instead of idling
                    llama1b::matvec_reduce<Config, pipeline::SCRATCH_BYTES_PER_WARP>(
                        pipeline::get_output_start(s, output_stage), up_out);
                } else {
                    // gate iteration: up_out already cached in registers
                    int block_idx = inst.block_at(i / 2);
                    uint8_t *scratch = pipeline::get_output_start(s, output_stage);
                    kittens::sv_bf<16> &out_smem = *reinterpret_cast<kittens::sv_bf<16> *>(scratch);

                    kittens::rv_fl<16> gate_out, gate_scratch;
                    llama1b::matvec_reduce<Config, pipeline::SCRATCH_BYTES_PER_WARP>(scratch, gate_out);

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

                        int sub_idx = block_idx * pipeline::MATVEC_BLOCK_SIZE / N;
                        barrier_arrive<Config>(&g.barriers.raw_ptr[inst.barrier_base + sub_idx], 1);
                    }
                    kittens::warp::sync();
                }

                if ((i + 1) % 2 == 0) {
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        int stage = (output_stage - j + OPS) % OPS;
                        kittens::warp::arrive(pipeline::outputs_finished(s, stage));
                    }
                }
                output_stage = (output_stage + 1) % OPS;
            }

            kittens::warp::sync();
            if (kittens::warp::elect_leader()) {
                kittens::tma::store_async_wait();
                s.page_finish(pipeline::get_activation_page(s));
            }
        }
    };
};

} // namespace megakittens
