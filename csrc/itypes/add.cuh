#pragma once

#include "kittens.cuh"

namespace megakittens {

template <typename Config, typename Globals, int SRC0, int SRC1, int DST>
struct Add {
    static constexpr int NUM_USED_PAGES = 2;
    using tile_t = kittens::st<kittens::bf16, 128, 128>;

    __device__ static __forceinline__ kittens::semaphore &inputs_arrived(state_t<Config> &s) { return s.semaphores()[0]; }
    __device__ static __forceinline__ kittens::semaphore &output_arrived(state_t<Config> &s) { return s.semaphores()[1]; }

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int lid) {
            return (lid + (Config::NUM_PAGES - NUM_USED_PAGES)) % Config::NUM_PAGES;
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            if (kittens::warp::elect_leader()) {
                kittens::init_semaphore(inputs_arrived(s), 1);
                kittens::init_semaphore(output_arrived(s), 1);
            }
            return 2;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const auto &src0 = g.template gls<SRC0>();
            const auto &src1 = g.template gls<SRC1>();

            if (kittens::laneid() >= NUM_USED_PAGES && kittens::laneid() < Config::NUM_PAGES) {
                const int pid = s.lid_to_pid(kittens::laneid());
                s.page_wait(pid);
                s.page_finish(pid, Config::NUM_CONSUMER_WARPS);
            }

            if (kittens::warp::elect_leader()) {
                #pragma unroll
                for (int i = 0; i < NUM_USED_PAGES; i++)
                    s.page_wait(s.lid_to_pid(i));

                tile_t &p0 = *reinterpret_cast<tile_t*>(s.pages[s.lid_to_pid(0)].data);
                tile_t &p1 = *reinterpret_cast<tile_t*>(s.pages[s.lid_to_pid(1)].data);
                
                #pragma unroll
                for (int i = 0; i < instruction_t::MAX_SRC_BARRIERS; i++) {
                    const int target = instruction.src_barrier_targets[i];
                    if (target > 0) {
                        int bid = instruction.src_barriers[i];
                        barrier_wait<Config>(&g.barriers.raw_ptr[bid], target);
                    }
                }

                kittens::tma::expect_bytes(inputs_arrived(s), 2*sizeof(tile_t));
                kittens::tma::load_async(p0, src0, {instruction.indices[0], instruction.indices[1]}, inputs_arrived(s));
                kittens::tma::load_async(p1, src1, {instruction.indices[0], instruction.indices[1]}, inputs_arrived(s));
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
        using consumer_group = kittens::group<Config::NUM_CONSUMER_WARPS>;
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            kittens::wait(inputs_arrived(s), 0);

            tile_t &p0 = *reinterpret_cast<tile_t*>(s.pages[s.lid_to_pid(0)].data);
            tile_t &p1 = *reinterpret_cast<tile_t*>(s.pages[s.lid_to_pid(1)].data);

            kittens::rt_bf<16, 128> src0_reg, src1_reg;
            consumer_group::load(src0_reg, p0);
            consumer_group::load(src1_reg, p1);
            consumer_group::add(src0_reg, src0_reg, src1_reg);
            consumer_group::store(p0, src0_reg);
            consumer_group::sync(1);

            if (consumer_group::elect_leader()) {
                kittens::arrive(output_arrived(s));
                s.page_finish(s.lid_to_pid(1), Config::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            auto &dst = g.template gls<DST>();

            kittens::wait(output_arrived(s), 0);

            tile_t &result = *reinterpret_cast<tile_t*>(s.pages[s.lid_to_pid(0)].data);

            if (kittens::warp::elect_leader()) {
                kittens::tma::store_async(dst, result, {instruction.indices[0], instruction.indices[1]});
                kittens::tma::store_async_read_wait();
                s.page_finish(s.lid_to_pid(0), Config::NUM_CONSUMER_WARPS);
            }

            if (kittens::laneid() == 0) {
                #pragma unroll
                for (int i = 0; i < instruction_t::MAX_DST_BARRIERS; i++) {
                    int bid = instruction.dst_barriers[i];
                    if (bid != 0xFF) barrier_arrive<Config>(&g.barriers.raw_ptr[bid], 1);
                }
            }
        }
    };
};

} // namespace megakittens
