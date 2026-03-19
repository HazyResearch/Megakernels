#pragma once

#include "kittens.cuh"

namespace megakittens {

#define MAKE_WORKER(name)                                                                      \
template <typename Config, typename Globals>                                                   \
__device__ __forceinline__ void name##_loop(const Globals &g, megakittens::state<Config> &s) { \
    for (s.iter = 0, s.stage = 0; true; ++s.iter) {                                            \
        const int phasebit = (s.iter / Config::INSTRUCTION_PIPE_STAGES) & 0b1;                 \
        kittens::wait(s.instruction_arrived[s.stage], phasebit);                               \
        const int opcode = s.instruction_states[s.stage].instruction[0];                       \
        if (opcode == -1) break;                                                               \
        dispatch_op<WorkerType::name, void>(opcode, g, s);                                     \
        kittens::warp::sync();                                                                 \
        if (kittens::warp::elect_leader() == 0)                                                \
            kittens::arrive(s.instruction_finished[s.stage]);                                  \
        kittens::ring_advance<Config::INSTRUCTION_PIPE_STAGES>(s.stage);                       \
    }                                                                                          \
}

MAKE_WORKER(loader)
MAKE_WORKER(launcher)
MAKE_WORKER(consumer)
MAKE_WORKER(storer)

} // namespace megakittens
