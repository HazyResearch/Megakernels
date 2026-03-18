#pragma once

#include "kittens.cuh"

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

MAKE_WORKER(loader, detail::TIMING_EVENT_SPECIAL_LOADER_START, detail::TIMING_EVENT_SPECIAL_LOADER_END, 1)
MAKE_WORKER(launcher, detail::TIMING_EVENT_SPECIAL_LAUNCHER_START, detail::TIMING_EVENT_SPECIAL_LAUNCHER_END, 1)
MAKE_WORKER(consumer, detail::TIMING_EVENT_SPECIAL_CONSUMER_START, detail::TIMING_EVENT_SPECIAL_CONSUMER_END, config::NUM_CONSUMER_WARPS)
MAKE_WORKER(storer, detail::TIMING_EVENT_SPECIAL_STORER_START, detail::TIMING_EVENT_SPECIAL_STORER_END, 1)
