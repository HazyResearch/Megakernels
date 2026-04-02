#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "utils.cuh"
#include "itypes/llama1b/matvec_pipeline.cuh"

namespace megakittens {

// Pipelined projection + residual add for LLaMA 1B decode.
// Computes: output += weights[layer][start_block*16 : end_block*16, reduction_slice] @ input[reduction_slice]
//
// Used for both o_proj (square: 2048x2048) and down_proj (rect: 2048x8192 with reduction splitting).
// Output uses TMA store_add_async for atomic residual accumulation.
//
// Template params:
//   N    = reduction dimension handled per instruction (pipeline capacity, e.g. 2048)
//   SRC0 = input activations [total_reduction_dim]       bf16 (warp::load from global)
//   SRC1 = weights [layers, output_dim, reduction_dim]   bf16 (TMA pipeline)
//   DST  = output  [output_dim]                          bf16 (TMA store_add_async)
//
// Instruction indices:
//   [0] = layer_idx
//   [1] = start_block  (output row blocks, units of 16)
//   [2] = end_block
//   [3] = reduction_col_offset  (element offset into input/weight cols; 0 when no splitting)

template <typename Config, typename Globals, int N, int SRC0, int SRC1, int DST>
struct ProjResidual {

    struct parsed_instruction {
        int layer_idx, start_block_idx, end_block_idx, reduction_col_offset, iters;
        __device__ inline parsed_instruction(const instruction_t &instruction) {
            layer_idx            = instruction.indices[0];
            start_block_idx      = instruction.indices[1];
            end_block_idx        = instruction.indices[2];
            reduction_col_offset = instruction.indices[3];
            iters                = end_block_idx - start_block_idx;
        }
        __device__ inline parsed_instruction(state_t<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        // Load one iteration of weight tiles via TMA
        __device__ static inline void
        load_iter(state_t<Config> &s, const Globals &g, parsed_instruction &inst,
                  int iter, int col_idx,
                  kittens::st_bf<16, 512> &weight_chunk,
                  kittens::semaphore &sem) {
            int block_idx = inst.start_block_idx + iter;
            int col_tile  = inst.reduction_col_offset / 512 + col_idx;
            kittens::tma::load_async(weight_chunk, g.template gls<SRC1>(),
                                     {inst.layer_idx, block_idx, col_tile}, sem);
        }

        // Reduce partial results from all consumer warps, then atomic-add to output
        __device__ static inline void
        store(state_t<Config> &s, const Globals &g, parsed_instruction &inst,
              int output_idx, int output_stage) {
            int block_idx = inst.start_block_idx + output_idx;

            uint8_t *output_scratch = pipeline::get_output_start(s, output_stage);

            kittens::rv_fl<16> output_rv;
            llama1b::matvec_reduce<Config, pipeline::SCRATCH_BYTES_PER_WARP>(
                output_scratch, output_rv);

            // Reuse scratch as TMA staging area (reduce already read from it)
            kittens::sv_bf<16> &output_smem_bf =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch);

            kittens::warp::sync();
            kittens::warp::store(output_smem_bf, output_rv);
            kittens::warp::sync();

            if (kittens::warp::elect_leader()) {
                kittens::tma::store_add_async(g.template gls<DST>(), output_smem_bf, {block_idx});
                kittens::tma::store_async_read_wait();
            }
            kittens::warp::sync();
        }
    };

    using pipeline = llama1b::matvec_pipeline<
        Config, Globals, N, parsed_instruction, pipeline_specifics>;

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
            s.tensor_wait();
            if (kittens::warp::elect_leader()) s.tensor_finish();
        }
    };

    struct consumer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            constexpr int ELEMS_PER_WARP = N / Config::NUM_CONSUMER_WARPS;
            using sv_t = kittens::sv_bf<ELEMS_PER_WARP>;
            using rv_t = kittens::rv_fl<ELEMS_PER_WARP>;

            parsed_instruction inst{s};

            // Warp 0 waits for activation page + upstream data barriers
            if (kittens::warpid() == 0 && kittens::laneid() == 0) {
                s.page_wait(pipeline::get_activation_page(s));
                all_input_barrier_wait<Config>(g, s.instruction());
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);

            // Each warp loads its slice of the input vector from global memory
            sv_t &activations_smem = reinterpret_cast<sv_t *>(
                &pipeline::get_activations(s))[kittens::warpid()];
            kittens::warp::load(activations_smem, g.template gls<SRC0>(),
                kittens::coord<>{inst.reduction_col_offset +
                                 kittens::warpid() * ELEMS_PER_WARP});
            kittens::warp::sync();

            // Shared memory -> registers
            rv_t activations_vec;
            kittens::warp::load(activations_vec, activations_smem);
            kittens::warp::sync();

            // Run pipelined matvec (writes partial results to activation page scratch)
            pipeline::consumer_loop(s, g, activations_vec);
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::storer_loop(s, g);
            // storer_loop already called store_async_wait + page_finish.
            // Now signal downstream ops that our global writes are visible.
            if (kittens::warp::elect_leader())
                all_barrier_arrive<Config>(g, s.instruction());
        }
    };
};

} // namespace megakittens
