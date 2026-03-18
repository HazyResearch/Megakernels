#pragma once

#include "kittens.cuh"
#include "config.cuh"

namespace megakittens {

__device__ inline unsigned int get_smid() {
    unsigned int ret;
    asm volatile("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

__device__ inline unsigned int get_worker_id() {
    return get_smid();
}

// Constants for logging. None of these should be seen by the user.
namespace detail {
constexpr int TIMING_EVENT_SPECIAL_OPCODE             = 0; // Stored by controller
constexpr int TIMING_EVENT_SPECIAL_WORKER_ID          = 1; // Stored by controller
constexpr int TIMING_EVENT_SPECIAL_CONTROLLER_START   = 5;
constexpr int TIMING_EVENT_SPECIAL_CONTROLLER_READY   = 6;
constexpr int TIMING_EVENT_SPECIAL_CONTROLLER_CLEANUP = 7;
constexpr int TIMING_EVENT_SPECIAL_LOADER_START       = 8;
constexpr int TIMING_EVENT_SPECIAL_LOADER_END         = 9;
constexpr int TIMING_EVENT_SPECIAL_LAUNCHER_START     = 10;
constexpr int TIMING_EVENT_SPECIAL_LAUNCHER_END       = 11;
constexpr int TIMING_EVENT_SPECIAL_CONSUMER_START     = 12;
constexpr int TIMING_EVENT_SPECIAL_CONSUMER_END       = 13;
constexpr int TIMING_EVENT_SPECIAL_STORER_START       = 14;
constexpr int TIMING_EVENT_SPECIAL_STORER_END         = 15;

constexpr int TIMING_EVENT_LOADER_REGION_START        = 16;
constexpr int TIMING_EVENT_LOADER_REGION_END          = 47;

constexpr int TIMING_EVENT_CONSUMER_REGION_START      = 48;
constexpr int TIMING_EVENT_CONSUMER_REGION_END        = 79;

constexpr int TIMING_EVENT_LAUNCHER_REGION_START      = 80;
constexpr int TIMING_EVENT_LAUNCHER_REGION_END        = 111;

constexpr int TIMING_EVENT_STORER_REGION_START        = 112;
constexpr int TIMING_EVENT_STORER_REGION_END          = 127;
}

// Constants for marking events.
constexpr int LOAD_EVENT     = 0;
constexpr int LOAD2_EVENT    = 1;
constexpr int COMPUTE_EVENT  = 2;
constexpr int COMPUTE2_EVENT = 3;
constexpr int COMPUTE3_EVENT = 4;
constexpr int STORE_EVENT    = 5;
constexpr int STORE2_EVENT   = 6;
constexpr int WAIT_EVENT     = 7;
constexpr int READY_EVENT    = 8;
constexpr int ERROR_EVENT    = 15;

__device__ inline uint64_t timestamp() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(ret) :: "memory");
    return ret;
}

template <template <typename> typename op_dispatcher, typename... ops>
struct dispatch_op {
    template <typename return_t, typename config, typename globals,
              typename... args>
    __device__ static inline return_t run(int opcode, const globals &g,
                                          args &...a) {
        // printf("Unknown opcode %d\n", opcode);
        asm volatile("trap;\n"); // we want to blow up in this case.
        return return_t{};
    } // do nothing, base case
};
template <template <typename> typename op_dispatcher, typename op,
          typename... ops>
struct dispatch_op<op_dispatcher, op, ops...> {
    template <typename return_t, typename config, typename globals,
              typename... args>
    __device__ static inline return_t run(int opcode, const globals &g,
                                          args &...a) {
        if (opcode == op::opcode)
            return op_dispatcher<op>::run(g, a...);
        else
            return dispatch_op<op_dispatcher, ops...>::template run<
                return_t, config, globals, args...>(opcode, g, a...);
    }
};

} // namespace megakittens

#ifdef MK_DEBUG
#define MK_DEBUG_PRINT_START(msg)                                              \
    printf("Thread %d: starting main loop for %s\n", threadIdx.x, msg);
#define MK_DEBUG_PRINT_END(msg)                                                \
    printf("Thread %d: exiting main loop for %s\n", threadIdx.x, msg);
#else
#define MK_DEBUG_PRINT_START(msg)
#define MK_DEBUG_PRINT_END(msg)
#endif

#define MAKE_WORKER(name, start_event, end_event, group_size)              \
namespace megakittens {                                                     \
namespace name {                                                           \
template <typename config, typename globals> struct name##_op_dispatcher { \
    template <typename op> struct dispatcher {                             \
        __device__ static inline void                                      \
        run(const globals &g, ::megakittens::state<config> &mks) {          \
            op::name::run(g, mks);                                         \
        }                                                                  \
    };                                                                     \
};                                                                         \
template <typename config, typename globals, typename... ops>              \
__device__ __forceinline__ void main_loop(const globals &g,                \
                            ::megakittens::state<config> &mks) {            \
    MK_DEBUG_PRINT_START(#name);                                           \
    int num_iters = g.instructions.rows();                                 \
    for (mks.instruction_index = 0, mks.instruction_ring = 0;              \
            mks.instruction_index < num_iters; mks.next_instruction()) {   // TODO: next_instruction no longer has syncwarp()
        mks.await_instruction();                                           \
        if (kittens::group<group_size>::laneid() == 0) {                   \
            mks.internal_record(start_event);                              \
        }                                                                  \
        if(mks.instruction()[0] == -1) break;                              \
        dispatch_op<name##_op_dispatcher<config, globals>::dispatcher,     \
                    ops...>::template run<void, config, globals,           \
                                            ::megakittens::state<config>>(  \
            mks.instruction()[0], g, mks);                                 \
        kittens::warp::sync();                                             \
        if (kittens::group<group_size>::laneid() == 0) {                   \
            mks.internal_record(end_event);                                \
        }                                                                  \
        mks.timing_event_offset = 0;                                       \
    }                                                                      \
    kittens::warp::sync();                                                 \
    MK_DEBUG_PRINT_END(#name);                                             \
}                                                                          \
}                                                                          \
}
