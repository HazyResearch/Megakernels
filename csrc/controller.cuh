#pragma once

#include "kittens.cuh"

namespace megakittens {

template <typename Config, typename Globals>
__device__ __forceinline__ void controller_loop(const Globals &g, megakittens::state_t<Config> &s) {
    const int cta_rank = ::kittens::cluster_ctarank();
    const int lane_id = ::kittens::laneid();
    int num_semaphores[Config::INSTRUCTION_PIPE_STAGES];
    int last_stage = -1;

    for (s.iter = 0, s.stage = 0; true; ++s.iter) {
        // Step 0. If this is not the first time the slot is being used, wait for the
        //         previous instruction to complete and invalidate its semaphores
        if (s.iter >= Config::INSTRUCTION_PIPE_STAGES) {
            const int phasebit = ((s.iter - Config::INSTRUCTION_PIPE_STAGES) / Config::INSTRUCTION_PIPE_STAGES) & 0b1;
            kittens::wait(s.instruction_finished[s.stage], phasebit);
            if (lane_id < num_semaphores[s.stage])
                invalidate_semaphore(s.instruction_states[s.stage].semaphores[lane_id]);
        }

        // Step 1. Query the CLC scheduler for the next instruction index
        int instruction_index;
        if (s.iter == 0) {
            instruction_index = blockIdx.x;
        } else {
            if (kittens::warp::elect_leader()) {
                if (cta_rank == 0)
                    kittens::clc::schedule(s.clc_handle[s.stage], s.clc_arrived[s.stage]);
                kittens::tma::expect_bytes(s.clc_arrived[s.stage], sizeof(s.clc_handle[s.stage]));
            }
            const int phasebit = ((s.iter - 1) / Config::INSTRUCTION_PIPE_STAGES) & 0b1;
            kittens::wait(s.clc_arrived[s.stage], phasebit);
            auto schedule = kittens::clc::query(s.clc_handle[s.stage]);
            if (!schedule.success) instruction_index = 0x7FFFFFFF; // signal to stop
            else                   instruction_index = schedule.x; // we only use 1D grid
        }

        // Step 2. Load the specific instruction from global to shared memory
        if (instruction_index >= g.instructions.rows()) {
            if (kittens::warp::elect_leader()) {
                s.instruction_states[s.stage].instruction[0] = -1; // signal to stop.
                kittens::arrive(s.instruction_arrived[s.stage], 1);
            }
            break;
        }
        int *instruction_ptr = &g.instructions[{instruction_index, 0}];
        if (lane_id < Config::INSTRUCTION_WIDTH)
            s.instruction_states[s.stage].instruction[lane_id] = instruction_ptr[lane_id];
        kittens::warp::sync();

        // Step 3. Establish physical page order
        if (s.iter == 0) {
            if (lane_id < Config::NUM_PAGES)
                s.instruction_states[s.stage].pid_order[lane_id] = lane_id;
        } else {
            const int last_opcode = s.instruction_states[last_stage].instruction[0];
            if (lane_id < Config::NUM_PAGES) {
                const int lid = dispatch_op<WorkerType::page_manager, int, Config, Globals>(
                        last_opcode, g, s, lane_id);
                s.instruction_states[s.stage].pid_order[lane_id] = s.instruction_states[last_stage].pid_order[lid];
            }
        }

        // Step 4. Initialize dynamic semaphores
        const int opcode = s.instruction_states[s.stage].instruction[0];
        num_semaphores[s.stage] = dispatch_op<WorkerType::semaphore_manager, int, Config, Globals>(opcode, g, s);
        asm volatile("{fence.proxy.async.shared::cta;\n}" ::: "memory"); // TODO: is this really needed?

        // Step 5. Signal other workers that the instruction/pages/semaphores are ready
        kittens::warp::sync();
        if (kittens::warp::elect_leader())
            kittens::arrive(s.instruction_arrived[s.stage], 1);

        // Update bookkeeping variables
        last_stage = s.stage;
        s.stage = kittens::ring_advance<Config::INSTRUCTION_PIPE_STAGES>(s.stage);
    }
}

} // namespace megakittens
