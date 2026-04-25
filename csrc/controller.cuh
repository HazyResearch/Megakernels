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
            kittens::warp::sync(); // invalidate_semaphore relies on the instruction
        }

        // Step 1. Fetch next instruction (mode-dependent)
        int *inst_src;
        if constexpr (Config::GLOBAL_WORK_QUEUE) {
            int instruction_index;
            if (s.iter == 0) {
                instruction_index = blockIdx.x;
            } else {
                const int phasebit = ((s.iter - 1) / Config::INSTRUCTION_PIPE_STAGES) & 0b1;
                if (kittens::warp::elect_leader()) {
                    if (cta_rank == 0) kittens::clc::schedule(s.clc_handle[s.stage], s.clc_arrived[s.stage]);
                    kittens::tma::expect_bytes(s.clc_arrived[s.stage], sizeof(s.clc_handle[s.stage]));
                }
                kittens::wait(s.clc_arrived[s.stage], phasebit);
                auto schedule = kittens::clc::query(s.clc_handle[s.stage]);
                if (!schedule.success) instruction_index = 0x7FFFFFFF; // signal to stop
                else                   instruction_index = schedule.x + cta_rank; // we only use 1D grid
            }
            if (instruction_index >= g.instructions.rows()) {
                if (kittens::warp::elect_leader()) {
                    s.instruction_states[s.stage].instruction.icode = -1; // signal to stop.
                    kittens::tensor_commit<Config::CLUSTER_SIZE>(s.instruction_arrived[s.stage]); // hack: use tcgen05.commit for mbarrier broadcast
                }
                break;
            }
            inst_src = &g.instructions[{instruction_index, 0}];
        } else {
            if ((int)s.iter >= g.instructions.depth()) {
                if (kittens::warp::elect_leader()) {
                    s.instruction_states[s.stage].instruction.icode = -1; // signal to stop.
                    kittens::tensor_commit<Config::CLUSTER_SIZE>(s.instruction_arrived[s.stage]); // hack: use tcgen05.commit for mbarrier broadcast
                }
                break;
            }
            inst_src = &g.instructions[{(int)s.iter, (int)blockIdx.x, 0}];
        }
        static_assert(sizeof(instruction_t)/sizeof(int) == 64); // 2 warp-wide loads
        int *inst_dst = reinterpret_cast<int*>(&s.instruction_states[s.stage].instruction);
        inst_dst[lane_id + 0]  = inst_src[lane_id + 0];
        inst_dst[lane_id + 32] = inst_src[lane_id + 32];
        kittens::warp::sync();

        // Step 2. Establish physical page order
        if (s.iter == 0) {
            if (lane_id < Config::NUM_PAGES)
                s.instruction_states[s.stage].pid_order[lane_id] = lane_id;
        } else if constexpr (Config::NO_VIRTUAL_SMEM || Config::INSTRUCTION_PIPE_STAGES == 1) {
            // Identity order was initialized once. With one pipe stage, the previous
            // instruction is fully drained before reuse, so virtual remapping is unnecessary.
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
        num_semaphores[s.stage] = dispatch_instruction<WorkerType::semaphore_manager, int, Config, Globals>(icode, g, s);
        asm volatile("{fence.proxy.async.shared::cta;\n}" ::: "memory"); // TODO: is this really needed?

        // Step 4. Signal other workers that the instruction/pages/semaphores are ready
        kittens::warp::sync();
        if (kittens::warp::elect_leader()) {
            if constexpr (Config::NO_INST_OVERLAP || Config::NO_INTER_OP_INST_OVERLAP) {
                if (s.iter > 0) {
                    const int phasebit = ((s.iter - 1) / Config::INSTRUCTION_PIPE_STAGES) & 0b1;
                    if constexpr (Config::NO_INST_OVERLAP) {
                        kittens::wait(s.instruction_finished[last_stage], phasebit);
                    } else {
                        const int last_icode = s.instruction_states[last_stage].instruction.icode;
                        if (icode != last_icode) kittens::wait(s.instruction_finished[last_stage], phasebit);
                    }
                }
            }
            kittens::tensor_commit<Config::CLUSTER_SIZE>(s.instruction_arrived[s.stage]); // hack: use tcgen05.commit for mbarrier broadcast
        }

        // Update bookkeeping variables
        last_stage = s.stage;
        s.stage = kittens::ring_advance<Config::INSTRUCTION_PIPE_STAGES>(s.stage);
    }
}

} // namespace megakittens
