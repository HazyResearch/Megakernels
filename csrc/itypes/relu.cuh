#pragma once

#include "kittens.cuh"

namespace megakittens {

template <typename Config, typename Globals, int SRC, int DST>
struct Relu {
    static constexpr int NUM_USED_PAGES = Config::NUM_PAGES; // 1 page per tile

    using tile_t = kittens::st<kittens::bf16, 128, 128>;

    __device__ static __forceinline__ kittens::semaphore &inputs_arrived(state_t<Config> &s, int i) { return s.semaphores()[i]; }

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            const int num_tiles = s.instruction().indices[2];
            const int num_unused = Config::NUM_PAGES - num_tiles;
            if (query < num_unused)
                return num_tiles + query;
            else
                return query - num_unused;
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            const int num_tiles = s.instruction().indices[2];
            if (kittens::laneid() < num_tiles)
                kittens::init_semaphore(inputs_arrived(s, kittens::laneid()), 1);
            return num_tiles;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const auto &src_gl = g.template gls<SRC>();
            const int tile_row = instruction.indices[0];
            const int tile_col_start = instruction.indices[1];
            const int num_tiles = instruction.indices[2];

            if (kittens::warp::elect_leader()) {
                all_barrier_wait<Config>(g, instruction);

                for (int i = 0; i < num_tiles; i++) {
                    const int pid = s.lid_to_pid(i);
                    s.page_wait(pid);
                    tile_t &src_smem = s.pages[pid].template as<tile_t>();
                    kittens::tma::expect_bytes(inputs_arrived(s, i), sizeof(tile_t));
                    kittens::tma::load_async(src_smem, src_gl, {tile_row, tile_col_start + i}, inputs_arrived(s, i));
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

        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            auto &dst_gl = g.template gls<DST>();
            const int tile_row = instruction.indices[0];
            const int tile_col_start = instruction.indices[1];
            const int num_tiles = instruction.indices[2];

            for (int t = 0; t < num_tiles; t++) {
                tile_t &src_smem = s.pages[s.lid_to_pid(t)].template as<tile_t>();
                kittens::wait(inputs_arrived(s, t), 0);

                kittens::rt_bf<16, 128> src_reg;
                consumer_group::load(src_reg, src_smem);
                consumer_group::relu(src_reg, src_reg);
                consumer_group::store(src_smem, src_reg);
                consumer_group::sync(1);

                if (consumer_group::elect_leader())
                    kittens::tma::store_async(dst_gl, src_smem, {tile_row, tile_col_start + t});
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
