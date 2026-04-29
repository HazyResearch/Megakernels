// TODO: very naive version. Must optimize

#pragma once

#include "kittens.cuh"

namespace megakittens {

enum class ReduceOp { MEAN };

template <typename Config, typename Globals, typename ElemType, ReduceOp Op, int SRC, int DST>
struct Reduce {
    static constexpr int TILE_ROWS = 128;

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
                const int pid = s.lid_to_pid(kittens::laneid());
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
        using consumer_group = kittens::group<Config::NUM_CONSUMER_WARPS>;

        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const auto &src_gl = g.template gls<SRC>();
            auto &dst_gl = g.template gls<DST>();

            const int src_batch = instruction.indices[0];
            const int src_depth = instruction.indices[1];
            const int src_row_start = instruction.indices[2];
            const int src_col_start = instruction.indices[3];
            const int dst_batch = instruction.indices[4];
            const int dst_depth = instruction.indices[5];
            const int dst_row = instruction.indices[6];
            const int dst_col_start = instruction.indices[7];
            const int num_rows = instruction.indices[8];
            const int num_cols = instruction.indices[9];

            if (consumer_group::elect_leader()) all_input_barrier_wait<Config>(g, instruction);
            consumer_group::sync(1);

            for (int row_offset = consumer_group::laneid(); row_offset < num_rows; row_offset += consumer_group::GROUP_THREADS) {
                float acc = 0.0f;
                #pragma unroll 1
                for (int col = 0; col < num_cols; col++) {
                    acc += static_cast<float>(src_gl[{src_batch, src_depth, src_row_start + row_offset, src_col_start + col}]);
                }

                if constexpr (Op == ReduceOp::MEAN) {
                    dst_gl[{dst_batch, dst_depth, dst_row, dst_col_start + row_offset}] = static_cast<ElemType>(acc / static_cast<float>(num_cols));
                } else {
                    static_assert(Op == ReduceOp::MEAN, "Unsupported ReduceOp");
                }
            }

            __threadfence();
            consumer_group::sync(1);
            if (consumer_group::elect_leader()) {
                all_reuse_barrier_wait<Config>(g, instruction);
                all_barrier_arrive<Config>(g, instruction);
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) { }
    };
};

} // namespace megakittens
