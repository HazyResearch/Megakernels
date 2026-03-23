#pragma once

#include "kittens.cuh"

namespace megakittens {

template <typename Config, typename Globals, int SRC0, int SRC1, int DST>
struct Add {
    static constexpr int TILE = 128; // must match Python TILE_SIZE
    static constexpr int NUM_USED_PAGES = 2;
    using tile_t = kittens::st<kittens::bf16, 128, 128, false>; // non-swizzled, 32KB = 1 page

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
            auto &src0 = g.template gls<SRC0>();
            auto &src1 = g.template gls<SRC1>();

            int pids[NUM_USED_PAGES];
            #pragma unroll
            for (int i = 0; i < NUM_USED_PAGES; i++) {
                pids[i] = s.lid_to_pid(i);
                s.page_wait(pids[i]);
            }

            if (kittens::laneid() == 0) {
                // Wait for cross-SM source barriers
                #pragma unroll
                for (int i = 0; i < instruction_t::MAX_SRC_BARRIERS; i++) {
                    int target = inst.src_barrier_targets[i];
                    if (target > 0) {
                        int bid = inst.src_barriers[i];
                        while (atomicAdd(&g.barriers.raw_ptr[bid], 0) < target)
                            nanosleep<Config::SPIN_LOOP_SLEEP_NS>();
                    }
                }

                tile_t &p0 = *reinterpret_cast<tile_t*>(s.pages[pids[0]].data);
                tile_t &p1 = *reinterpret_cast<tile_t*>(s.pages[pids[1]].data);

                kittens::tma::expect_bytes(inputs_arrived(s), 2 * sizeof(tile_t));
                kittens::tma::load_async(p0, src0, {inst.indices[0], inst.indices[1]}, inputs_arrived(s));
                kittens::tma::load_async(p1, src1, {inst.indices[0], inst.indices[1]}, inputs_arrived(s));
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

            // Each warpgroup handles half the tile (64 rows)
            int gid = kittens::warpgroup::groupid();
            using half_t = kittens::st<kittens::bf16, 64, 128, false>;

            half_t &src0_half = reinterpret_cast<half_t*>(s.pages[s.lid_to_pid(0)].data)[gid];
            half_t &src1_half = reinterpret_cast<half_t*>(s.pages[s.lid_to_pid(1)].data)[gid];

            kittens::rt_bf<16, 128> src0_reg, src1_reg;
            kittens::warpgroup::load(src0_reg, src0_half);
            kittens::warpgroup::load(src1_reg, src1_half);
            kittens::warpgroup::add(src0_reg, src0_reg, src1_reg);
            kittens::warpgroup::store(src0_half, src0_reg);

            kittens::warpgroup::sync(gid);
            if (kittens::warp::elect_leader()) kittens::arrive(output_arrived(s));
            // Release src1 page
            if (kittens::warpid() == 0 && kittens::warp::elect_leader())
                s.page_finish(s.lid_to_pid(1), Config::NUM_CONSUMER_WARPS);
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &inst = s.instruction();
            auto &dst = g.template gls<DST>();

            kittens::wait(output_arrived(s), 0);

            tile_t &result = *reinterpret_cast<tile_t*>(s.pages[s.lid_to_pid(0)].data);

            if (kittens::laneid() == 0) {
                kittens::tma::store_async(dst, result, {inst.indices[0], inst.indices[1]});
                kittens::tma::store_async_wait();
            }
            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                s.page_finish(s.lid_to_pid(0), Config::NUM_CONSUMER_WARPS);

                __threadfence();
                #pragma unroll
                for (int i = 0; i < instruction_t::MAX_DST_BARRIERS; i++) {
                    int bid = inst.dst_barriers[i];
                    if (bid != 0xFF) atomicAdd(&g.barriers.raw_ptr[bid], 1);
                }
            }
        }
    };
};

} // namespace megakittens
