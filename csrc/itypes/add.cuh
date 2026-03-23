#pragma once

#include "kittens.cuh"

namespace megakittens {

// Element-wise add using TMA for global↔shared transfers.
//   Pages 0,1: src0 tile (upper/lower halves, overwritten with result)
//   Pages 2,3: src1 tile (upper/lower halves)
// Each page holds a 64×128 bf16 sub-tile (16KB = 1 page).
// The scheduler tile is 128×128, split into two 64×128 TMA transfers.
// Semaphores: [0] inputs_arrived (loader → consumer), [1] output_arrived (consumer → storer)
template <typename Config, typename Globals, int SRC0, int SRC1, int DST>
struct Add {
    static constexpr int TILE = 128; // must match Python tile_size
    static constexpr int NUM_USED_PAGES = 4;
    using tile_t = kittens::st<kittens::bf16, 64, 128, false>; // non-swizzled, 16KB = 1 page

    __device__ static inline kittens::semaphore &inputs_arrived(state_t<Config> &s) { return s.semaphores()[0]; }
    __device__ static inline kittens::semaphore &output_arrived(state_t<Config> &s) { return s.semaphores()[1]; }

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int lid) {
            return (lid + (Config::NUM_PAGES - NUM_USED_PAGES)) % Config::NUM_PAGES;
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            if (kittens::laneid() == 0) {
                kittens::init_semaphore(inputs_arrived(s), 1);
                kittens::init_semaphore(output_arrived(s), Config::NUM_CONSUMER_WARPS);
            }
            return 2;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &inst = s.instruction();
            const int tile_i = inst[7], tile_j = inst[8];
            auto &src0 = g.template gls<SRC0>();
            auto &src1 = g.template gls<SRC1>();

            int pids[NUM_USED_PAGES];
            #pragma unroll
            for (int i = 0; i < NUM_USED_PAGES; i++) {
                pids[i] = s.lid_to_pid(i);
                s.page_wait(pids[i]);
            }

            if (kittens::laneid() == 0) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int target = inst[23 + i];
                    if (target > 0) {
                        int bid = (inst[21 + i/4] >> ((i%4)*8)) & 0xFF;
                        while (atomicAdd(&g.barriers.raw_ptr[bid], 0) < target)
                            asm volatile("nanosleep.u32 %0;\n" :: "r"(20));
                    }
                }

                tile_t &p0 = *reinterpret_cast<tile_t*>(s.pages[pids[0]].data);
                tile_t &p1 = *reinterpret_cast<tile_t*>(s.pages[pids[1]].data);
                tile_t &p2 = *reinterpret_cast<tile_t*>(s.pages[pids[2]].data);
                tile_t &p3 = *reinterpret_cast<tile_t*>(s.pages[pids[3]].data);

                kittens::tma::expect_bytes(inputs_arrived(s), 4 * sizeof(tile_t));
                kittens::tma::load_async(p0, src0, {tile_i*2,     tile_j}, inputs_arrived(s));
                kittens::tma::load_async(p1, src0, {tile_i*2 + 1, tile_j}, inputs_arrived(s));
                kittens::tma::load_async(p2, src1, {tile_i*2,     tile_j}, inputs_arrived(s));
                kittens::tma::load_async(p3, src1, {tile_i*2 + 1, tile_j}, inputs_arrived(s));
            }

            if (kittens::laneid() >= NUM_USED_PAGES && kittens::laneid() < Config::NUM_PAGES) {
                int pid = s.lid_to_pid(kittens::laneid());
                s.page_wait(pid);
                s.page_finish(pid, Config::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct launcher {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            s.tensor_wait();
            if (kittens::warp::elect_leader()) s.tensor_finish(Config::NUM_CONSUMER_WARPS);
        }
    };

    struct consumer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            kittens::wait(inputs_arrived(s), 0);

            int gid = kittens::warpgroup::groupid();
            tile_t &src0_tile = *reinterpret_cast<tile_t*>(s.pages[s.lid_to_pid(gid)].data);
            tile_t &src1_tile = *reinterpret_cast<tile_t*>(s.pages[s.lid_to_pid(gid + 2)].data);

            kittens::rt_bf<16, 128> src0_reg, src1_reg;
            kittens::warpgroup::load(src0_reg, src0_tile);
            kittens::warpgroup::load(src1_reg, src1_tile);
            kittens::warpgroup::add(src0_reg, src0_reg, src1_reg);
            kittens::warpgroup::store(src0_tile, src0_reg);

            kittens::warpgroup::sync(gid);
            if (kittens::warp::elect_leader()) kittens::arrive(output_arrived(s));
            if (kittens::warpid() == 0 && kittens::warp::elect_leader()) {
                s.page_finish(s.lid_to_pid(2), Config::NUM_CONSUMER_WARPS);
                s.page_finish(s.lid_to_pid(3), Config::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &inst = s.instruction();
            const int tile_i = inst[7], tile_j = inst[8];
            auto &dst = g.template gls<DST>();

            kittens::wait(output_arrived(s), 0);

            tile_t &result_upper = *reinterpret_cast<tile_t*>(s.pages[s.lid_to_pid(0)].data);
            tile_t &result_lower = *reinterpret_cast<tile_t*>(s.pages[s.lid_to_pid(1)].data);

            if (kittens::laneid() == 0) {
                kittens::tma::store_async(dst, result_upper, {tile_i*2,     tile_j});
                kittens::tma::store_async(dst, result_lower, {tile_i*2 + 1, tile_j});
                kittens::tma::store_async_wait();
            }
            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                s.page_finish(s.lid_to_pid(0), Config::NUM_CONSUMER_WARPS);
                s.page_finish(s.lid_to_pid(1), Config::NUM_CONSUMER_WARPS);

                __threadfence();
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int bid = (inst[31] >> (i*8)) & 0xFF;
                    if (bid != 0xFF) atomicAdd(&g.barriers.raw_ptr[bid], 1);
                }
            }
        }
    };
};

} // namespace megakittens
