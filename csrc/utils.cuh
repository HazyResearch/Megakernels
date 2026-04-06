#pragma once

#include "kittens.cuh"

namespace megakittens {

#pragma nv_diag_suppress 940
template <MegaKittensIType Op, WorkerType worker_type, typename T, typename... Args>
__device__ __forceinline__ static T dispatch_instruction(Args &...args) {
    if constexpr      (worker_type == WorkerType::page_manager)      return Op::controller::lid_release_order(args...);
    else if constexpr (worker_type == WorkerType::semaphore_manager) return Op::controller::init_semaphores(args...);
    else if constexpr (worker_type == WorkerType::loader)            return Op::loader::run(args...);
    else if constexpr (worker_type == WorkerType::launcher)          return Op::launcher::run(args...);
    else if constexpr (worker_type == WorkerType::consumer)          return Op::consumer::run(args...);
    else if constexpr (worker_type == WorkerType::storer)            return Op::storer::run(args...);
    else static_assert(sizeof(T) == 9999, "Invalid WorkerType");
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
__device__ __forceinline__ void all_input_barrier_wait(const Globals &g, const instruction_t &inst) {
    for (int i = 0; i < inst.num_input_barriers; i++)
        barrier_wait<Config>(&g.barriers.raw_ptr[inst.src_barriers[i]], inst.src_barrier_targets[i]);
}

template <typename Config, typename Globals>
__device__ __forceinline__ void all_reuse_barrier_wait(const Globals &g, const instruction_t &inst) {
    for (int i = inst.num_input_barriers; i < inst.num_input_barriers + inst.num_reuse_barriers; i++)
        barrier_wait<Config>(&g.barriers.raw_ptr[inst.src_barriers[i]], inst.src_barrier_targets[i]);
}

template <typename Config, typename Globals>
__device__ __forceinline__ void all_barrier_arrive(const Globals &g, const instruction_t &inst) {
    for (int i = 0; i < inst.num_dst_barriers; i++)
        barrier_arrive<Config>(&g.barriers.raw_ptr[inst.dst_barriers[i]], 1);
}

// timing event constants for profiling 
// slot 0 is for icode
constexpr int TEVENT_ICODE           = 0;
constexpr int TEVENT_AT_GMEM_WAIT    = 1;
constexpr int TEVENT_DONE_GMEM_WAIT  = 2;
constexpr int TEVENT_FIRST_LOAD      = 3;
constexpr int TEVENT_LAST_LOAD       = 4;
constexpr int TEVENT_FIRST_USE       = 5;
constexpr int TEVENT_LAST_USE        = 6;
constexpr int TEVENT_FIRST_STORE     = 7;
constexpr int TEVENT_LAST_STORE      = 8;
constexpr int TEVENT_CONSUMER_START  = 9;
constexpr int TEVENT_OUTPUT_READY    = 10;
constexpr int TEVENT_AT_CTRL_WAIT    = 11;
constexpr int TEVENT_DONE_CTRL_WAIT  = 12;

} // namespace megakittens
