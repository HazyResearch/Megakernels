#pragma once

#include "kittens.cuh"

namespace megakittens {

template <typename Config, typename Globals>
__device__ __forceinline__ void controller_loop(const Globals &g, megakittens::state_t<Config> &s) {
    const int lane_id = ::kittens::laneid();
    int num_semaphores[Config::INSTRUCTION_PIPE_STAGES];
    int last_stage = -1;
    const unsigned int num_iters = g.instructions.rows();

    for (s.iter = 0, s.stage = 0; s.iter < num_iters; ++s.iter) {
        // Step 0. If this is not the first time the slot is being used, wait for the
        //         previous instruction to complete and invalidate its semaphores
        if (s.iter >= Config::INSTRUCTION_PIPE_STAGES) {
            const int phasebit = ((s.iter - Config::INSTRUCTION_PIPE_STAGES) / Config::INSTRUCTION_PIPE_STAGES) & 0b1;
            if (lane_id == 0) s.record(TEVENT_AT_CTRL_WAIT);
            kittens::wait(s.instruction_finished[s.stage], phasebit);
            if (lane_id == 0) s.record(TEVENT_DONE_CTRL_WAIT);
            if (lane_id < num_semaphores[s.stage])
                invalidate_semaphore(s.instruction_states[s.stage].semaphores[lane_id]);
            kittens::warp::sync();
        }

        // Step 1. Load instruction from per-SM queue
        static_assert(sizeof(instruction_t)/sizeof(int) == 64); // 2 warp-wide loads
        int *inst_src = &g.instructions[{(int)blockIdx.x, (int)s.iter, 0}];
        int *inst_dst = reinterpret_cast<int*>(&s.instruction_states[s.stage].instruction);
        inst_dst[lane_id + 0]  = inst_src[lane_id + 0];
        inst_dst[lane_id + 32] = inst_src[lane_id + 32];
        kittens::warp::sync();

        // Step 2. Establish physical page order
        if (s.iter == 0) {
            if (lane_id < Config::NUM_PAGES)
                s.instruction_states[s.stage].pid_order[lane_id] = lane_id;
        } else {
            const int last_icode = s.instruction_states[last_stage].instruction.icode;
            if (lane_id < Config::NUM_PAGES) {
                const uint32_t current_stage = s.stage;
                s.stage = last_stage; // so lid_release_order(...) can use s.instruction()
                const int lid = dispatch_instruction<WorkerType::page_manager, int, Config, Globals>(last_icode, g, s, lane_id);
                s.stage = current_stage;
                s.instruction_states[s.stage].pid_order[lane_id] = s.instruction_states[last_stage].pid_order[lid];
            }
        }

        // Step 3. Initialize dynamic semaphores
        const int icode = s.instruction_states[s.stage].instruction.icode;
        if (lane_id == 0) s.record(TEVENT_ICODE, icode);
        if (icode == 0) {
            num_semaphores[s.stage] = 0;
        } else {
            num_semaphores[s.stage] = dispatch_instruction<WorkerType::semaphore_manager, int, Config, Globals>(icode, g, s);
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        // Step 4. Signal other workers that the instruction/pages/semaphores are ready
        if (lane_id == 0)
            kittens::arrive(s.instruction_arrived[s.stage]);

        // Update bookkeeping variables
        last_stage = s.stage;
        s.stage = kittens::ring_advance<Config::INSTRUCTION_PIPE_STAGES>(s.stage);
    }
}

} // namespace megakittens
