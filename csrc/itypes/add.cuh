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
            const auto &a_gl = g.template gls<SRC0>();
            const auto &b_gl = g.template gls<SRC1>();

            if (kittens::laneid() >= NUM_USED_PAGES && kittens::laneid() < Config::NUM_PAGES) {
                const int pid = s.lid_to_pid(kittens::laneid());
                s.page_wait(pid);
                s.page_finish(pid);
            }

            if (kittens::warp::elect_leader()) {
                #pragma unroll
                for (int i = 0; i < NUM_USED_PAGES; i++)
                    s.page_wait(s.lid_to_pid(i));

                tile_t &a_smem = s.pages[s.lid_to_pid(0)].template as<tile_t>();
                tile_t &b_smem = s.pages[s.lid_to_pid(1)].template as<tile_t>();
                
                all_barrier_wait<Config>(g, instruction);

                kittens::tma::expect_bytes(inputs_arrived(s), 2*sizeof(tile_t));
                kittens::tma::load_async(a_smem, a_gl, {instruction.indices[0], instruction.indices[1]}, inputs_arrived(s));
                kittens::tma::load_async(b_smem, b_gl, {instruction.indices[0], instruction.indices[1]}, inputs_arrived(s));
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
            kittens::wait(inputs_arrived(s), 0);

            tile_t &a_smem = s.pages[s.lid_to_pid(0)].template as<tile_t>();
            tile_t &b_smem = s.pages[s.lid_to_pid(1)].template as<tile_t>();

            kittens::rt_bf<16, 128> a_reg, b_reg;
            consumer_group::load(a_reg, a_smem);
            consumer_group::load(b_reg, b_smem);
            consumer_group::add(a_reg, a_reg, b_reg);
            consumer_group::store(a_smem, a_reg); // reuse
            consumer_group::sync(1);

            if (consumer_group::elect_leader()) {
                kittens::arrive(output_arrived(s));
                s.page_finish(s.lid_to_pid(1));
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            auto &c_gl = g.template gls<DST>();

            kittens::wait(output_arrived(s), 0);

            tile_t &c_smem = s.pages[s.lid_to_pid(0)].template as<tile_t>();

            if (kittens::warp::elect_leader()) {
                kittens::tma::store_async(c_gl, c_smem, {instruction.indices[0], instruction.indices[1]});
                kittens::tma::store_async_wait();
                s.page_finish(s.lid_to_pid(0));
                all_barrier_arrive<Config>(g, instruction);
            }
        }
    };
};

} // namespace megakittens
