#pragma once

#include "kittens.cuh"

namespace megakittens {

template <typename Config, typename Globals, int SRC_A, int SRC_B, int DST>
struct Add {
    static constexpr int TILES_PER_INST = 3;
    static constexpr int NUM_USED_PAGES = TILES_PER_INST*2;

    using tile_t = kittens::st<kittens::bf16, 128, 128>;

    __device__ static __forceinline__ kittens::semaphore &inputs_arrived(state_t<Config> &s, int i) { return s.semaphores()[i]; }

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            const int num_tiles = s.instruction().indices[2];
            const int num_unused = Config::NUM_PAGES - num_tiles*2;
            if (query < num_unused) // Unused pages
                return num_tiles*2 + query;
            else if (query < num_unused + num_tiles) // B pages
                return (query - num_unused)*2 + 1;
            else // A pages
                return (query - num_unused - num_tiles)*2;
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            if (kittens::laneid() < TILES_PER_INST)
                kittens::init_semaphore(inputs_arrived(s, kittens::laneid()), 1);
            return TILES_PER_INST;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const auto &a_gl = g.template gls<SRC_A>();
            const auto &b_gl = g.template gls<SRC_B>();
            const int tile_row = instruction.indices[0];
            const int tile_col_start = instruction.indices[1];
            const int num_tiles = instruction.indices[2];

            if (kittens::warp::elect_leader()) {
                all_input_barrier_wait<Config>(g, instruction);

                for (int i = 0; i < num_tiles; i++) {
                    const int a_pid = s.lid_to_pid(i*2);
                    const int b_pid = s.lid_to_pid(i*2 + 1);
                    s.page_wait(a_pid);
                    s.page_wait(b_pid);
                    tile_t &a_st = s.pages[a_pid].template as<tile_t>();
                    tile_t &b_st = s.pages[b_pid].template as<tile_t>();
                    kittens::tma::expect_bytes(inputs_arrived(s, i), 2 * sizeof(tile_t));
                    kittens::tma::load_async(a_st, a_gl, {tile_row, tile_col_start + i}, inputs_arrived(s, i));
                    kittens::tma::load_async(b_st, b_gl, {tile_row, tile_col_start + i}, inputs_arrived(s, i));
                }
            } else if (kittens::warp::elect_leader_from_active()) {
                for (int i = num_tiles*2; i < Config::NUM_PAGES; i++) {
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
            auto &c_gl = g.template gls<DST>();
            const int tile_row = instruction.indices[0];
            const int tile_col_start = instruction.indices[1];
            const int num_tiles = instruction.indices[2];

            for (int t = 0; t < num_tiles; t++) {
                kittens::wait(inputs_arrived(s, t), 0);

                tile_t &a_st = s.pages[s.lid_to_pid(t*2)].template as<tile_t>();
                tile_t &b_st = s.pages[s.lid_to_pid(t*2 + 1)].template as<tile_t>();

                kittens::rt_bf<16, 128> a_reg, b_reg;
                consumer_group::load(a_reg, a_st);
                consumer_group::load(b_reg, b_st);
                consumer_group::add(a_reg, a_reg, b_reg);
                consumer_group::store(a_st, a_reg);
                consumer_group::sync(1);

                if (consumer_group::elect_leader()) {
                    s.page_finish(s.lid_to_pid(t*2 + 1)); // release B page
                    if (t == 0) all_reuse_barrier_wait<Config>(g, instruction);
                    kittens::tma::store_async(c_gl, a_st, {tile_row, tile_col_start + t});
                }
            }

            // Wait for all TMA stores and release A pages
            if (consumer_group::elect_leader()) {
                kittens::tma::store_async_wait();
                for (int t = 0; t < num_tiles; t++) s.page_finish(s.lid_to_pid(t*2));
                all_barrier_arrive<Config>(g, instruction);
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) { }
    };
};

} // namespace megakittens
