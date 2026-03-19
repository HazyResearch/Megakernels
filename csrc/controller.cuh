#pragma once

#include "kittens.cuh"

namespace megakittens {
namespace controller {

template <typename Config, typename Globals>
struct page_allocator_op_dispatcher {
    template <typename op> struct dispatcher {
        __device__ static inline int
        run(const Globals &g, typename Config::instruction_t &instruction,
            int &query) {
            return op::controller::release_lid(g, instruction, query);
        }
    };
};


template <typename Config, typename Globals>
struct semaphore_constructor_op_dispatcher {
    template <typename op> struct dispatcher {
        __device__ static inline int
        run(const Globals &g, ::megakittens::state<Config> &s) {
            auto out = op::controller::init_semaphores(g, s);
            asm volatile("{fence.proxy.async.shared::cta;\n}" ::: "memory");
            return out;
        }
    };
};


template <typename Config, typename Globals, typename... ops>
__device__ void main_loop(const Globals &g, megakittens::state_t<Config> &s) {
    const int cta_rank = ::kittens::cluster_ctarank();
    const int lane_id = ::kittens::laneid();
    int num_semaphores[Config::INSTRUCTION_PIPE_STAGES];
    int stage = 0;
    int last_stage = -1;

    for (int i = 0; true; ++i) {
        // Step 0. If this is not the first time the slot is being used, wait for the
        //         previous instruction to complete and invalidate its semaphores
        if (i >= Config::INSTRUCTION_PIPE_STAGES) {
            int phasebit = ((i - Config::INSTRUCTION_PIPE_STAGES) / Config::INSTRUCTION_PIPE_STAGES) & 0b1;
            kittens::wait(s.instruction_finished[stage], phasebit);
            if (lane_id < num_semaphores[stage])
                invalidate_semaphore(s.instruction_states[stage].semaphores[lane_id]);
        }

        // Step 1. Query the CLC scheduler for the next instruction index
        int instruction_index;
        if (i == 0) {
            instruction_index = blockIdx.x;
        } else {
            if (warp::elect_leader()) {
                if (cta_rank == 0)
                    kittens::clc::schedule(s.clc_handle[stage], s.clc_arrived[stage]);
                kittens::tma::expect_bytes(s.clc_arrived[stage], sizeof(s.clc_handle[stage]));
            }
            int phasebit = ((i - 1) / Config::INSTRUCTION_PIPE_STAGES) & 0b1;
            kittens::wait(s.clc_arrived[stage], get_phasebit<0>(phasebits, stage));
            auto schedule = kittens::clc::query(s.clc_handle[stage]);
            if (!schedule.success) instruction_index = 0x7FFFFFFF; // signal to stop
            else                   instruction_index = schedule.x; // we only use 1D grid
        }

        // Step 2. Load the specific instruction from global to shared memory
        if (instruction_index >= g.instructions.rows()) {
            if (warp::elect_leader()) {
                s.instruction_states[stage].instruction[0] = -1; // signal to stop.
                kittens::arrive(s.instruction_arrived[stage], 1);
            }
            break;
        }
        int *instruction_ptr = &g.instructions[{instruction_index, 0}];
        if (lane_id < Config::INSTRUCTION_WIDTH)
            s.instruction_states[stage].instruction[lane_id] = instruction_ptr[lane_id];

        // Step 3. Establish physical page order
        if (i == 0) {
            if (lane_id < Config::NUM_PAGES) 
                s.instruction_states[stage].pid_order[lane_id] = lane_id;
        } else {
            int last_opcode = s.instruction_states[last_stage].instructions[0];
            if (lane_id < Config::NUM_PAGES) {
                int lid = dispatch_op<page_allocator_op_dispatcher<Config, Globals>::dispatcher, ops...>::template run<int, Config, Globals, Config::instruction_t, int>(
                        last_opcode, g, s.instruction_states[last_stage].instructions, lane_id); // todo fix
                s.instruction_states[stage].pid_order[lane_id] = s.instruction_states[last_stage].pid_order[lid];
            }
        }

        // Step 4. Initialize dynamic semaphores
        int opcode = s.instruction()[0];
        if (opcode == 0) {
            num_semaphores[s.instruction_ring] = 0; // todo just remove; add it to no-op
        } else {
            num_semaphores[s.instruction_ring] = dispatch_op<
                semaphore_constructor_op_dispatcher<Config, Globals>::dispatcher, ops...>::template run<int, Config, Globals, ::megakittens::state<Config>>(
                    opcode, g, s); // todo fix + are we sure this is same op regardless of lane id?
        }

        // Step 5. Signal other workers that the instruction/pages/semaphores are ready
        if (warp::elect_leader())
            kittens::arrive(s.instruction_arrived[stage], 1);

        // Update bookkeeping variables
        last_stage = stage;
        ring_advance<Config::INSTRUCTION_PIPE_STAGES>(stage);
        kittens::warp::sync();
    }
}

} // namespace controller
} // namespace megakittens
