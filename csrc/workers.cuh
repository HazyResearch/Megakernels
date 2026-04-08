#pragma once

#include "kittens.cuh"

namespace megakittens {

#define MAKE_WORKER(name)                                                                        \
template <typename Config, typename Globals>                                                     \
__device__ __forceinline__ void name##_loop(const Globals &g, megakittens::state_t<Config> &s) { \
    const unsigned int num_iters = g.instructions.rows();                                          \
    for (s.iter = 0, s.stage = 0; s.iter < num_iters; ++s.iter) {                                \
        const int phasebit = (s.iter / Config::INSTRUCTION_PIPE_STAGES) & 0b1;                   \
        kittens::wait(s.instruction_arrived[s.stage], phasebit);                                 \
        dispatch_instruction<WorkerType::name, void, Config, Globals>(                           \
            s.instruction_states[s.stage].instruction.icode, g, s);                              \
        kittens::warp::sync();                                                                   \
        if (kittens::warp::elect_leader())                                                       \
            kittens::arrive(s.instruction_finished[s.stage]);                                    \
        s.stage = kittens::ring_advance<Config::INSTRUCTION_PIPE_STAGES>(s.stage);               \
    }                                                                                            \
}

MAKE_WORKER(loader)
MAKE_WORKER(launcher)
MAKE_WORKER(consumer)
MAKE_WORKER(storer)

} // namespace megakittens
