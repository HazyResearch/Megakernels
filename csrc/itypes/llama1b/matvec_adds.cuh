#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "utils.cuh"
#include "itypes/llama1b/matvec_pipeline.cuh"

namespace megakittens {

template <typename Config, typename Globals, int N, int SRC0, int SRC1, int DST>
struct MatVecAdds {

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
        __device__ static inline void
        load_iter(state_t<Config> &s, const Globals &g, parsed_instruction &inst,
                  int iter, int col_idx,
                  kittens::st_bf<16, 512> &weight_chunk,
                  kittens::semaphore &sem) {
            int block_idx = inst.start_block_idx + iter;
            int col_tile  = inst.reduction_col_offset / 512 + col_idx;
            kittens::tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                                     weight_chunk, g.template gls<SRC1>(),
                                     {inst.layer_idx, block_idx, col_tile}, sem);
        }

        __device__ static inline void
        store(state_t<Config> &s, const Globals &g, parsed_instruction &inst,
              int output_idx, int output_stage) {
            int block_idx = inst.start_block_idx + output_idx;

            uint8_t *output_scratch = pipeline::get_output_start(s, output_stage);

            kittens::rv_fl<16> output_rv;
            llama1b::matvec_reduce<Config, pipeline::SCRATCH_BYTES_PER_WARP>(
                output_scratch, output_rv);

            kittens::sv_bf<16> &output_smem_bf =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch);

            kittens::warp::sync();
            kittens::warp::store(output_smem_bf, output_rv);
            kittens::warp::sync();

            if (kittens::warp::elect_leader()) {
                kittens::tma::store_add_async<kittens::cache_policy::EVICT_LAST>(g.template gls<DST>(), output_smem_bf, {0, block_idx});
                kittens::tma::store_async_read_wait();
                // atomic add here
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
            pipeline::launcher_loop(s, g);
        }
    };

    struct consumer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            constexpr int ELEMS_PER_WARP = N / Config::NUM_CONSUMER_WARPS;
            using sv_t = kittens::sv_bf<ELEMS_PER_WARP>;
            using rv_t = kittens::rv_fl<ELEMS_PER_WARP>;

            parsed_instruction inst{s};

            if (kittens::warpid() == 0 && kittens::laneid() == 0) {
                s.page_wait(pipeline::get_activation_page(s));
                all_input_barrier_wait<Config>(g, s.instruction());
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);

            // please see if we can load instead of ptr crap here
            sv_t &activations_smem = reinterpret_cast<sv_t *>(
                &pipeline::get_activations(s))[kittens::warpid()];
            {
                const kittens::bf16 *src = reinterpret_cast<const kittens::bf16 *>(
                    g.template gls<SRC0>().raw_ptr)
                    + inst.reduction_col_offset + kittens::warpid() * ELEMS_PER_WARP;
                #pragma unroll
                for (int i = kittens::laneid(); i < ELEMS_PER_WARP; i += kittens::WARP_THREADS)
                    activations_smem.data[i] = src[i];
            }
            kittens::warp::sync();

            rv_t activations_vec;
            kittens::warp::load(activations_vec, activations_smem);
            kittens::warp::sync();

            pipeline::consumer_loop(s, g, activations_vec);
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::storer_loop(s, g);
            if (kittens::warp::elect_leader()) {
                __threadfence();
                all_barrier_arrive<Config>(g, s.instruction());
            }
        }
    };
};

} // namespace megakittens
