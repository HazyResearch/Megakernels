#include "kittens.cuh"

#include "op_op.cuh"

namespace megakittens {

enum WorkerType {
    page_manager = 0,
    semaphore_manager = 1,
    loader = 2,
    launcher = 3,
    consumer = 4,
    storer = 5
};

template <typename Op, WorkerType worker_type, typename T, typename Globals, typename... Args>
__device__ __forceinline__ static T dispatch_op(const Globals &g, Args &...args) {
    if constexpr      (worker_type == WorkerType::page_manager)      return Op::controller::release_lid(g, args...);
    else if constexpr (worker_type == WorkerType::semaphore_manager) return Op::controller::init_semaphores(g, args...);
    else if constexpr (worker_type == WorkerType::loader)            return Op::loader::run(g, args...);
    else if constexpr (worker_type == WorkerType::launcher)          return Op::launcher::run(g, args...);
    else if constexpr (worker_type == WorkerType::consumer)          return Op::consumer::run(g, args...);
    else if constexpr (worker_type == WorkerType::storer)            return Op::storer::run(g, args...);
    else asm volatile("{trap;\n}");
}

template <WorkerType worker_type, typename T, typename Globals, typename... Args>
__device__ __forceinline__ static T dispatch_op(const int opcode, const Globals &g, Args &...args) {
    switch (opcode) {
        case 0:
            return dispatch_op<NoOp, worker_type, T>(g, args...);
            break;
        default:
            asm volatile("{trap;\n}");
    }
}

} // namespace megakittens
