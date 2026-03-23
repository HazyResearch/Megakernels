#pragma once

#include "kittens.cuh"

namespace megakittens {

template <int ns>
__device__ __forceinline__ static void nanosleep() {
    static_assert(ns <= 1000000, "nanosleep duration exceeds 1ms");
    asm volatile("{nanosleep.u32 %0;}" :: "r"(ns));
}

template <MegaKittensIType Op, WorkerType worker_type, typename T, typename... Args>
__device__ __forceinline__ static T dispatch_instruction(Args &...args) {
    if constexpr      (worker_type == WorkerType::page_manager)      return Op::controller::lid_release_order(args...);
    else if constexpr (worker_type == WorkerType::semaphore_manager) return Op::controller::init_semaphores(args...);
    else if constexpr (worker_type == WorkerType::loader)            return Op::loader::run(args...);
    else if constexpr (worker_type == WorkerType::launcher)          return Op::launcher::run(args...);
    else if constexpr (worker_type == WorkerType::consumer)          return Op::consumer::run(args...);
    else if constexpr (worker_type == WorkerType::storer)            return Op::storer::run(args...);
    else asm volatile("{trap;\n}");
}

} // namespace megakittens
