#pragma once

#include "kittens.cuh"

#include "../util.cuh"

namespace megakernel {
namespace controller {

template <typename config, typename globals>
struct semaphore_constructor_op_dispatcher {
    template <typename op> struct dispatcher {
        __device__ static inline int
        run(const globals &g, ::megakernel::state<config> &kvms) {
            auto out = op::controller::init_semaphores(g, kvms);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            return out;
        }
    };
};

template <typename config, typename globals, typename... ops>
__device__ void inline semaphore_constructor_loop(
    const globals &g, ::megakernel::state<config> &kvms) {
    static_assert(config::INSTRUCTION_PIPELINE_STAGES == 2,
                  "Need to be changed.");
    int num_iters = g.instructions.rows();
    int tic = 0;
    int last_num_semaphores;
    for (kvms.instruction_index = 0, kvms.instruction_ring = 0;
         kvms.instruction_index < num_iters;
         kvms.instruction_index++,
        kvms.instruction_ring =
             ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(
                 kvms.instruction_ring),
        tic = 1 - tic) {

        kittens::wait(kvms.instruction_arrived[kvms.instruction_ring],
             (kvms.instruction_index / config::INSTRUCTION_PIPELINE_STAGES) &
                 1);
        int opcode = kvms.instruction()[0];
        int next_num_semaphores;
        if (opcode == 0) {
            next_num_semaphores = 0;
        } else {
            next_num_semaphores = dispatch_op<
                semaphore_constructor_op_dispatcher<config,
                                                    globals>::dispatcher,
                ops...>::template run<int, config, globals,
                                      ::megakernel::state<config>>(
                opcode, g, kvms);
        }
        arrive(kvms.semaphores_ready);
        if (kvms.instruction_index > 0) {
            int last_ring = ring_retreat<config::INSTRUCTION_PIPELINE_STAGES>(
                kvms.instruction_ring);
            kittens::wait(kvms.instruction_finished[last_ring],
                 ((kvms.instruction_index - 1) /
                  config::INSTRUCTION_PIPELINE_STAGES) &
                     1);
            for (int i = 0; i < last_num_semaphores; i++) {
                invalidate_semaphore(
                    kvms.all_instructions[last_ring].semaphores[i]);
            }
        }
        last_num_semaphores = next_num_semaphores;
    }
    // if(blockIdx.x == 0) printf("110\n");
    if (num_iters > 0) {
        int last_ring = ring_retreat<config::INSTRUCTION_PIPELINE_STAGES>(
            kvms.instruction_ring);
        kittens::wait(kvms.instruction_finished[last_ring],
             ((kvms.instruction_index - 1) /
              config::INSTRUCTION_PIPELINE_STAGES) &
                 1);
        for (int i = 0; i < last_num_semaphores; i++) {
            invalidate_semaphore(
                kvms.all_instructions[last_ring].semaphores[i]);
        }
    }
}

} // namespace controller
} // namespace megakernel
