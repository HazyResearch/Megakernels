#pragma once

#include "kittens.cuh"

namespace megakittens {

template <typename Config, typename Globals, typename SrcDType, typename DstDType, int SRC, int DST>
struct Copy {
    static constexpr int MAX_DTYPE_SIZE = sizeof(SrcDType) > sizeof(DstDType) ? sizeof(SrcDType) : sizeof(DstDType);
    static constexpr int TILE_ROWS = 128;
    static constexpr int TILE_COLS = Config::PAGE_SIZE / (TILE_ROWS * MAX_DTYPE_SIZE);

    using src_tile_t = kittens::st<SrcDType, TILE_ROWS, TILE_COLS>;
    using dst_tile_t = kittens::st<DstDType, TILE_ROWS, TILE_COLS>;

    __device__ static __forceinline__ kittens::semaphore &inputs_arrived(state_t<Config> &s, int i) { return s.semaphores()[i]; }

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            const int num_tiles = s.instruction().indices[8];
            const int num_unused = Config::NUM_PAGES - num_tiles;
            if (query < num_unused)
                return num_tiles + query;
            else
                return query - num_unused;
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            const int num_tiles = s.instruction().indices[8];
            if (kittens::laneid() < num_tiles)
                kittens::init_semaphore(inputs_arrived(s, kittens::laneid()), 1);
            return num_tiles;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const auto &src_gl = g.template gls<SRC>();
            const int src_batch = instruction.indices[0];
            const int src_depth = instruction.indices[1];
            const int src_tile_row = instruction.indices[2];
            const int src_tile_col_start = instruction.indices[3];
            const int num_tiles = instruction.indices[8];

            if (kittens::warp::elect_leader()) {
                all_input_barrier_wait<Config>(g, instruction);

                for (int i = 0; i < num_tiles; i++) {
                    const int pid = s.lid_to_pid(i);
                    s.page_wait(pid);
                    src_tile_t &src_smem = s.pages[pid].template as<src_tile_t>();
                    kittens::tma::expect_bytes(inputs_arrived(s, i), sizeof(src_tile_t));
                    kittens::tma::load_async(src_smem, src_gl, {src_batch, src_depth, src_tile_row, src_tile_col_start + i}, inputs_arrived(s, i));
                }
            } else if (kittens::warp::elect_leader_from_active()) {
                for (int i = num_tiles; i < Config::NUM_PAGES; i++) {
                    s.page_wait(s.lid_to_pid(i));
                    s.page_finish(s.lid_to_pid(i));
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
        using consumer_group = kittens::group<Config::NUM_CONSUMER_WARPS>;
        static constexpr int REG_HEIGHT = TILE_ROWS / Config::NUM_CONSUMER_WARPS;

        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            auto &dst_gl = g.template gls<DST>();
            const int dst_batch = instruction.indices[4];
            const int dst_depth = instruction.indices[5];
            const int dst_tile_row = instruction.indices[6];
            const int dst_tile_col_start = instruction.indices[7];
            const int num_tiles = instruction.indices[8];

            for (int t = 0; t < num_tiles; t++) {
                const int pid = s.lid_to_pid(t);
                src_tile_t &src_smem = s.pages[pid].template as<src_tile_t>();
                kittens::wait(inputs_arrived(s, t), 0);

                kittens::rt<SrcDType, REG_HEIGHT, TILE_COLS> src_reg;
                consumer_group::load(src_reg, src_smem);

                kittens::rt<DstDType, REG_HEIGHT, TILE_COLS> dst_reg;
                consumer_group::copy(dst_reg, src_reg);

                dst_tile_t &dst_smem = s.pages[pid].template as<dst_tile_t>();
                consumer_group::store(dst_smem, dst_reg);
                consumer_group::sync(1);

                if (consumer_group::elect_leader()) {
                    if (t == 0) all_reuse_barrier_wait<Config>(g, instruction);
                    kittens::tma::store_async(dst_gl, dst_smem, {dst_batch, dst_depth, dst_tile_row, dst_tile_col_start + t});
                }
            }

            if (consumer_group::elect_leader()) {
                kittens::tma::store_async_wait();
                for (int t = 0; t < num_tiles; t++) s.page_finish(s.lid_to_pid(t));
                all_barrier_arrive<Config>(g, instruction);
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) { }
    };
};

} // namespace megakittens
