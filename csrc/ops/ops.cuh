#pragma once

#include "kittens.cuh"

#include "no_op.cuh"

namespace megakittens {

enum WorkerType {
    page_manager = 0,
    semaphore_manager = 1,
    loader = 2,
    launcher = 3,
    consumer = 4,
    storer = 5
};

template <typename Op, WorkerType worker_type, typename T, typename... Args>
__device__ __forceinline__ static T dispatch_op(Args &...args) {
    if constexpr      (worker_type == WorkerType::page_manager)      return Op::controller::lid_release_order(args...);
    else if constexpr (worker_type == WorkerType::semaphore_manager) return Op::controller::init_semaphores(args...);
    else if constexpr (worker_type == WorkerType::loader)            return Op::loader::run(args...);
    else if constexpr (worker_type == WorkerType::launcher)          return Op::launcher::run(args...);
    else if constexpr (worker_type == WorkerType::consumer)          return Op::consumer::run(args...);
    else if constexpr (worker_type == WorkerType::storer)            return Op::storer::run(args...);
    else asm volatile("{trap;\n}");
}

template <WorkerType worker_type, typename T, typename Config, typename Globals, typename... Args>
__device__ __forceinline__ static T dispatch_op(const int opcode, Args &...args) {
    switch (opcode) {
        case 0:
            return dispatch_op<NoOp<Config, Globals>, worker_type, T>(args...);
            break;
        default:
            return dispatch_op<NoOp<Config, Globals>, worker_type, T>(args...);
            // TODO: revert to:
            // asm volatile("{trap;\n}");
    }
}

} // namespace megakittens
