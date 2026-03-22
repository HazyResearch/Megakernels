#pragma once

#include "kittens.cuh"

namespace megakittens {

// Element-wise add using shared memory pages.
//   Page 0: src0 tile (overwritten in-place with result)
//   Page 1: src1 tile
// Semaphores: [0] inputs_arrived (loader → consumer), [1] output_arrived (consumer → storer)
template <typename Config, typename Globals, int SRC0, int SRC1, int DST>
struct Add {
    static constexpr int TILE = 64;
    static constexpr int NUM_USED_PAGES = 2;

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

            const int pid0 = s.lid_to_pid(0), pid1 = s.lid_to_pid(1);
            const int tile_rows = min(TILE, (int)src0.rows() - tile_i * TILE);
            const int tile_cols = min(TILE, (int)src0.cols() - tile_j * TILE);
            const int total = tile_rows * tile_cols;

            // Wait for cross-SM source barriers before loading
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
            }

            // Claim pages and load src tiles from global → smem
            s.page_wait(pid0);
            s.page_wait(pid1);
            kittens::bf16 *smem0 = (kittens::bf16*)s.pages[pid0].data;
            kittens::bf16 *smem1 = (kittens::bf16*)s.pages[pid1].data;
            for (int i = kittens::laneid(); i < total; i += kittens::WARP_THREADS) {
                int r = i / tile_cols, c = i % tile_cols;
                smem0[i] = src0.raw_ptr[(tile_i * TILE + r) * src0.cols() + tile_j * TILE + c];
            }
            for (int i = kittens::laneid(); i < total; i += kittens::WARP_THREADS) {
                int r = i / tile_cols, c = i % tile_cols;
                smem1[i] = src1.raw_ptr[(tile_i * TILE + r) * src1.cols() + tile_j * TILE + c];
            }
            kittens::warp::sync();
            if (kittens::laneid() == 0) kittens::arrive(inputs_arrived(s));

            // Release unused pages
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
            const auto &inst = s.instruction();
            const int tile_i = inst[7], tile_j = inst[8];
            auto &dst = g.template gls<DST>();

            const int tile_rows = min(TILE, (int)dst.rows() - tile_i * TILE);
            const int tile_cols = min(TILE, (int)dst.cols() - tile_j * TILE);
            const int total = tile_rows * tile_cols;

            kittens::wait(inputs_arrived(s), 0);

            kittens::bf16 *smem0 = (kittens::bf16*)s.pages[s.lid_to_pid(0)].data;
            kittens::bf16 *smem1 = (kittens::bf16*)s.pages[s.lid_to_pid(1)].data;

            // All consumer threads add in-place: smem0[i] = smem0[i] + smem1[i]
            for (int i = threadIdx.x; i < total; i += Config::NUM_CONSUMER_WARPS * kittens::WARP_THREADS) {
                smem0[i] = __float2bfloat16(__bfloat162float(smem0[i]) + __bfloat162float(smem1[i]));
            }

            __syncwarp();
            if (kittens::warp::elect_leader()) kittens::arrive(output_arrived(s));
            // Release src1 page
            if (kittens::warpid() == 0 && kittens::warp::elect_leader())
                s.page_finish(s.lid_to_pid(1), Config::NUM_CONSUMER_WARPS);
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &inst = s.instruction();
            const int tile_i = inst[7], tile_j = inst[8];
            auto &dst = g.template gls<DST>();

            const int tile_rows = min(TILE, (int)dst.rows() - tile_i * TILE);
            const int tile_cols = min(TILE, (int)dst.cols() - tile_j * TILE);
            const int total = tile_rows * tile_cols;

            kittens::wait(output_arrived(s), 0);

            // Store result from page 0 → global
            kittens::bf16 *smem0 = (kittens::bf16*)s.pages[s.lid_to_pid(0)].data;
            for (int i = kittens::laneid(); i < total; i += kittens::WARP_THREADS) {
                int r = i / tile_cols, c = i % tile_cols;
                dst.raw_ptr[(tile_i * TILE + r) * dst.cols() + tile_j * TILE + c] = smem0[i];
            }
            kittens::warp::sync();
            if (kittens::laneid() == 0) {
                s.page_finish(s.lid_to_pid(0), Config::NUM_CONSUMER_WARPS);

                // Signal cross-SM destination barriers (0xFF = unused)
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
