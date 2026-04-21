#pragma once

#include "kittens.cuh"
#include "utils.cuh"

namespace megakittens {

// STATUS: ported — untested end-to-end (needs the 70B decode loop harness).
//
// BarrierInc — LLaMA-70B cross-SM barrier pre-credit (reference opcode 12,
// matching `inc_barriers.cu::inc_barriers_op`).
//
// The reference pre-credits a fixed set of barrier slots — one per layer for
// AttnNorm / MlpNorm / GQA_AttentionDecode / LM_HeadNorm — so ops whose
// batch-padding slots never produce work don't deadlock their downstream
// consumers. The framework version is simpler: the scheduler emits the
// specific (barrier_id, amount) pairs to pre-credit as instruction indices,
// and this itype atomically adds each amount to its barrier.
//
// Instruction indices:
//   indices[0]           = N (number of barriers to credit)
//   indices[1 + 2*i]     = barrier_id (index into g.barriers)
//   indices[2 + 2*i]     = amount (int32, added atomically)
//
// MAX_INDICES = 16 → up to 7 pairs per instruction; emit multiple
// instructions if the layer loop needs more.
template <typename Config, typename Globals>
struct BarrierInc {
    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int lid) {
            return lid;
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            return 0;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            // No page memory used — release immediately so the next instruction
            // can claim our pages.
            if (kittens::laneid() < Config::NUM_PAGES) {
                const int pid = s.lid_to_pid(kittens::laneid());
                s.page_wait(pid);
                s.page_finish(pid);
            }

            if (kittens::warp::elect_leader()) {
                const auto &inst = s.instruction();
                all_input_barrier_wait<Config>(g, inst);

                const int n = inst.indices[0];
                int *bar_base = g.barriers.raw_ptr;
                #pragma unroll 1
                for (int i = 0; i < n; i++) {
                    const int bid    = inst.indices[1 + 2 * i];
                    const int amount = inst.indices[2 + 2 * i];
                    atomicAdd(bar_base + bid, amount);
                }

                all_barrier_arrive<Config>(g, inst);
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
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {}
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {}
    };
};

} // namespace megakittens
