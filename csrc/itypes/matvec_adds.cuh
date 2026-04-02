#pragma once

#include "kittens.cuh"

namespace megakittens {

// Projection + residual add for decode.
// Computes: output[row] += dot(weights[layer][row], input) for each row in [start_block*16, end_block*16).
//
// Works for both o_proj (square: 2048x2048) and down_proj (rectangular: 2048x8192).
//
// indices[0] = layer_idx
// indices[1] = start_block (in units of 16 output rows)
// indices[2] = end_block
//
// SRC0 = input        [reduction_dim]                         bf16  (e.g. attn_out or silu_out)
// SRC1 = weights      [NUM_LAYERS, output_dim, reduction_dim] bf16
// DST  = output       [output_dim]                            bf16  (read-modify-write)

template <typename Config, typename Globals, int SRC0, int SRC1, int DST>
struct MatVecAdds {
    static constexpr int BLOCK_SIZE = 16;

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int lid) {
            if (lid < Config::NUM_PAGES - 1) return lid + 1;
            return 0;
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            return 0;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            if (kittens::warp::elect_leader()) {
                for (int i = 0; i < Config::NUM_PAGES; i++) {
                    int pid = s.lid_to_pid(i);
                    s.page_wait(pid);
                    s.page_finish(pid);
                }
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
        __device__ __forceinline__ static float warp_reduce_sum(float val) {
            #pragma unroll
            for (int offset = kittens::WARP_THREADS / 2; offset > 0; offset >>= 1)
                val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
            return val;
        }

        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            if (kittens::group<Config::NUM_CONSUMER_WARPS>::elect_leader())
                all_input_barrier_wait<Config>(g, s.instruction());
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);

            const auto &inst = s.instruction();
            const int layer_idx   = inst.indices[0];
            const int start_block = inst.indices[1];
            const int end_block   = inst.indices[2];
            const int num_rows    = (end_block - start_block) * BLOCK_SIZE;
            const int row_base    = start_block * BLOCK_SIZE;
            const int warp_id     = kittens::warpid();
            const int lane        = kittens::laneid();

            // Tensor pointers
            const kittens::bf16 *input   = reinterpret_cast<const kittens::bf16 *>(g.template gls<SRC0>().raw_ptr);
            const kittens::bf16 *weights = reinterpret_cast<const kittens::bf16 *>(g.template gls<SRC1>().raw_ptr);
            kittens::bf16 *output        = reinterpret_cast<kittens::bf16 *>(g.template gls<DST>().raw_ptr);

            // reduction_dim = length of input vector (2048 for o_proj, 8192 for down_proj)
            const int reduction_dim = g.template gls<SRC0>().cols();
            // output_dim = length of output vector (2048 for both)
            const int output_dim = g.template gls<DST>().cols();

            // Weights layout: [num_layers, output_dim, reduction_dim]
            const int layer_offset = layer_idx * output_dim * reduction_dim;

            // Distribute rows across consumer warps
            const int rows_per_warp = (num_rows + Config::NUM_CONSUMER_WARPS - 1) / Config::NUM_CONSUMER_WARPS;
            const int my_row_start  = warp_id * rows_per_warp;
            const int my_row_end    = min(my_row_start + rows_per_warp, num_rows);

            // Elements per thread for the dot product
            const int elems_per_thread = (reduction_dim + kittens::WARP_THREADS - 1) / kittens::WARP_THREADS;

            for (int local_row = my_row_start; local_row < my_row_end; local_row++) {
                const int global_row = row_base + local_row;
                const kittens::bf16 *w_row = weights + layer_offset + global_row * reduction_dim;

                // Dot product: w_row @ input
                float acc = 0.0f;
                for (int i = 0; i < elems_per_thread; i++) {
                    int col = lane + i * kittens::WARP_THREADS;
                    if (col < reduction_dim) {
                        acc += __bfloat162float(w_row[col]) * __bfloat162float(input[col]);
                    }
                }
                acc = warp_reduce_sum(acc);

                // Add to output (one thread writes)
                if (lane == 0) {
                    float old_val = __bfloat162float(output[global_row]);
                    output[global_row] = __float2bfloat16(old_val + acc);
                }
            }

            // Barrier arrive
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(1);
            if (kittens::group<Config::NUM_CONSUMER_WARPS>::elect_leader()) {
                __threadfence();
                all_barrier_arrive<Config>(g, s.instruction());
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {}
    };
};

} // namespace megakittens
