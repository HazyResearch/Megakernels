#pragma once

#include "kittens.cuh"
#include "utils.cuh"

namespace megakittens {

// STATUS: working — tested via `modal run scripts/modal_8gpu.py --which all-device-barrier`.
//
// AllDeviceBarrier — LLaMA-70B cross-device global sync (reference opcode
// 13, matching `csrc/itypes/reference/all_device_barrier.cu`).
//
// Each SM on every device atomically bumps a shared counter via
// `red.relaxed.sys.global.add` and then spin-waits until the counter
// reaches `gridDim.x * NUM_DEVICES`. All devices converge on device 0's
// physical slot of the PGL barrier (any single device works; picking 0
// matches the reference). SYS-scope atomics keep the counter coherent
// across NVLink peers.
//
// No tensor data, no TMA, no matmul — pure inter-SM/inter-device sync.
//
// Instruction indices:
//   indices[0] = slot_idx — which int32 in the shared barrier buffer.
//
// Template params:
//   SRC_BAR_PGL — input handle (unused by the kernel; exists because the
//                 scheduler requires the DAG to have at least one input
//                 tensor and because the barrier IType takes one).
//   DST_BAR_PGL — output PGL slot. Expected inner type `gl<int, 1, 1, 1, -1>`;
//                 framework-allocated (zero-initialized per device). All
//                 devices bump device 0's physical slot via P2P.
template <typename Config, typename Globals, int SRC_BAR_PGL, int DST_BAR_PGL>
struct AllDeviceBarrier {
    // SYS-scope primitives. Local to this itype so utils.cuh can keep its
    // GPU-scope defaults (see the `TODO: change scope to sys for multi-gpu`
    // comments there).
    __device__ __forceinline__ static void sys_redAdd(int *addr, int val) {
        asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;"
                     :: "l"(addr), "r"(val) : "memory");
    }

    __device__ __forceinline__ static void sys_wait(const int *addr, int target) {
        int val;
        do {
            asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                         : "=r"(val) : "l"(addr) : "memory");
        } while (val < target);
        asm volatile("fence.acquire.sys;" ::: "memory");
    }

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
            // Release every page immediately — this itype owns no page memory.
            if (kittens::laneid() < Config::NUM_PAGES) {
                const int pid = s.lid_to_pid(kittens::laneid());
                s.page_wait(pid);
                s.page_finish(pid);
            }

            if (kittens::warp::elect_leader()) {
                const auto &inst = s.instruction();

                // Respect scheduler-managed src barriers before entering the
                // cross-device rendezvous.
                all_input_barrier_wait<Config>(g, inst);

                const int slot_idx = inst.indices[0];
                const auto &bar_pgl = g.template pgls<DST_BAR_PGL>();
                int *bar_addr = bar_pgl.gls[0].raw_ptr + slot_idx;

                const int target = gridDim.x * Globals::NUM_DEVICES;
                sys_redAdd(bar_addr, 1);
                sys_wait(bar_addr, target);

                // Release scheduler-managed dst barriers for downstream
                // instructions on this device.
                all_barrier_arrive<Config>(g, inst);
            }
        }
    };

    struct launcher {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            // Blackwell's 2-stage tensor-allocator pipeline still expects
            // the handshake even when we produce no TMEM output.
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
