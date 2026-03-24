#pragma once

#include "kittens.cuh"

namespace megakittens {

template <MegaKittensIType Op, WorkerType worker_type, typename T, typename... Args>
__device__ __forceinline__ static T dispatch_instruction(Args &...args) {
    if constexpr      (worker_type == WorkerType::page_manager)      return Op::controller::lid_release_order(args...);
    else if constexpr (worker_type == WorkerType::semaphore_manager) return Op::controller::init_semaphores(args...);
    else if constexpr (worker_type == WorkerType::loader)            return Op::loader::run(args...);
    else if constexpr (worker_type == WorkerType::launcher)          return Op::launcher::run(args...);
    else if constexpr (worker_type == WorkerType::consumer)          return Op::consumer::run(args...);
    else if constexpr (worker_type == WorkerType::storer)            return Op::storer::run(args...);
    else { asm volatile("{trap;\n}"); return T{}; }
}

template <typename Config>
__device__ __forceinline__ static void nanosleep() {
    static_assert(Config::SPIN_LOOP_SLEEP_NS <= 1000000, "nanosleep duration exceeds 1ms");
    asm volatile("{nanosleep.u32 %0;}" :: "r"(Config::SPIN_LOOP_SLEEP_NS));
}

template <typename Config>
__device__ __forceinline__ void barrier_wait(int* barrier_addr, const int target) {
    int barrier_val;
    do {
        asm volatile("{ld.relaxed.gpu.global.u32 %0, [%1];}" // should not spin-loop with acquire
            : "=r"(barrier_val) : "l"(barrier_addr) : "memory"); // TODO: change scope to `sys` for multi-gpu setting
    } while (barrier_val != target);
    asm volatile("{fence.acquire.gpu;}" ::: "memory"); // TODO: change scope to `sys` for multi-gpu setting
}

template <typename Config>
__device__ __forceinline__ void barrier_arrive(int* barrier_addr, const int val) {
    asm volatile("{red.release.gpu.global.add.u32 [%0], %1;}" // TODO: change scope to `sys` for multi-gpu setting
        :: "l"(barrier_addr), "r"(val) : "memory");
}

template <typename Config, typename Globals>
__device__ __forceinline__ void all_barrier_wait(const Globals &g, const instruction_t &inst) {
    #pragma unroll
    for (int i = 0; i < instruction_t::MAX_SRC_BARRIERS; i++) {
        const int target = inst.src_barrier_targets[i];
        if (target > 0) barrier_wait<Config>(&g.barriers.raw_ptr[inst.src_barriers[i]], target);
    }
}

template <typename Config, typename Globals>
__device__ __forceinline__ void all_barrier_arrive(const Globals &g, const instruction_t &inst) {
    #pragma unroll
    for (int i = 0; i < instruction_t::MAX_DST_BARRIERS; i++) {
        int bid = inst.dst_barriers[i];
        if (bid != 0xFF) barrier_arrive<Config>(&g.barriers.raw_ptr[bid], 1);
    }
}

} // namespace megakittens
