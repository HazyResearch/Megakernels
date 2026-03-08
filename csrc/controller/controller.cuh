#pragma once

#include "kittens.cuh"

#include "../util.cuh"
#include "instruction_fetch.cuh"
#include "timings_store.cuh"
#include "semaphore_constructor.cuh"
#include "page_allocator.cuh"

namespace megakernel {
namespace controller {

template <typename config, typename globals, typename... ops>
__device__ void main_loop(const globals &g, ::megakernel::state<config> &kvms) {
    auto laneid = ::kittens::laneid();
    int num_iters = g.instructions.rows();
    int num_semaphores[config::INSTRUCTION_PIPELINE_STAGES];

    // for warps
    static_assert(config::NUM_PAGES <= 32);

    int last_global_instruction_indices[config::INSTRUCTION_PIPELINE_STAGES];
    for (kvms.instruction_index = 0, kvms.instruction_ring = 0;
         kvms.instruction_index < num_iters;
         kvms.instruction_index++,
        kvms.instruction_ring =
             ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(
                 kvms.instruction_ring)) {


        // Step 0. if the slot was used in the previous iteration, wait for the
        // previous instruction to complete & invalidate its semaphores
        if (kvms.instruction_index >= config::INSTRUCTION_PIPELINE_STAGES) {
            int last_slot_instruction_index =
                kvms.instruction_index - config::INSTRUCTION_PIPELINE_STAGES;

            int phasebit = (last_slot_instruction_index /
                            config::INSTRUCTION_PIPELINE_STAGES) & 1;
            kittens::wait(kvms.instruction_finished[kvms.instruction_ring], phasebit);

            if constexpr (config::TIMING_RECORD_ENABLED) {
                if (laneid == 0) kvms.internal_record(detail::TIMING_EVENT_SPECIAL_CONTROLLER_CLEANUP);
            }


            int num_to_invalidate = num_semaphores[kvms.instruction_ring];
            for (int sem_idx = laneid; sem_idx < num_to_invalidate; sem_idx += 32) {
                invalidate_semaphore(
                    kvms.all_instructions[kvms.instruction_ring]
                        .semaphores[sem_idx]);
            }

            // TODO needed?
            kittens::warp::sync();

            if constexpr (config::ENABLE_GLOBAL_WORK_QUEUE) {
                last_slot_instruction_index = last_global_instruction_indices[0];
            }

            if constexpr (config::TIMING_RECORD_ENABLED) {
                store_timings_and_reset<config, globals>(
                    &kvms.all_instructions[kvms.instruction_ring]
                            .timings[0],
                    last_slot_instruction_index, g);
            }
        }

        int global_instruction_index; // get the next global instruction index
        int start_time;
        if constexpr (config::ENABLE_GLOBAL_WORK_QUEUE) {
            wait(kvms.instruction_fetch_ready, (kvms.instruction_index%2)^1);
            if constexpr (config::TIMING_RECORD_ENABLED) {
                start_time = (int)(timestamp() - kvms.start_clock);
            }
            if (laneid == 0) global_instruction_index = atomicAdd(&g.global_instruction_index[{}], 1);
            global_instruction_index = __shfl_sync(0xffffffff, global_instruction_index, 0);
        } else {
            global_instruction_index = kvms.instruction_index;
        }

        if constexpr (config::TIMING_RECORD_ENABLED && !config::ENABLE_GLOBAL_WORK_QUEUE) {
            start_time = (int)(timestamp() - kvms.start_clock);
        }
        if constexpr (config::TIMING_RECORD_ENABLED) {
            if (laneid == 0) kvms.timing()[detail::TIMING_EVENT_SPECIAL_CONTROLLER_START] = start_time;
        }
        if(global_instruction_index >= g.instructions.rows() || !load_instructions<config, globals>(&kvms.instruction()[0],
                                           global_instruction_index, g)) {
            if(laneid == 0) {
                kvms.instruction()[0] = -1; // this is a signal to other warps to stop.
                arrive(kvms.instruction_arrived[kvms.instruction_ring], 1);
            }
            kittens::warp::sync();
            break;
        }

        // Step 2. Establish physical page order
        int last_instruction_ring =
            (kvms.instruction_ring + config::INSTRUCTION_PIPELINE_STAGES - 1) %
            config::INSTRUCTION_PIPELINE_STAGES;

        if (kvms.instruction_index == 0) {
            if (laneid < config::NUM_PAGES) {
                kvms.pid_order()[laneid] = laneid;
            }
        } else {
            auto last_opcode =
                kvms.all_instructions[last_instruction_ring].instructions[0];

            if (laneid < config::NUM_PAGES) {
                int lid = dispatch_op<
                    page_allocator_op_dispatcher<config, globals>::dispatcher,
                    ops...>::template run<int, config, globals,
                                          config::instruction_t, int>(
                    last_opcode, g,
                    kvms.all_instructions[last_instruction_ring].instructions,
                    laneid);

                kvms.pid_order()[laneid] =
                    kvms.all_instructions[last_instruction_ring].pid_order[lid];
            }
        }

        // Step 3. Construct semaphores
        int opcode = kvms.instruction()[0];
        int meta = opcode;
        if(laneid == 1) meta = get_worker_id(); // worker id
        if(laneid <= 1) kvms.timing()[laneid] = meta; // store meta data for instruction.
        if (opcode == 0) {
            num_semaphores[kvms.instruction_ring] = 0;
        } else {
            num_semaphores[kvms.instruction_ring] = dispatch_op<
                semaphore_constructor_op_dispatcher<config,
                                                    globals>::dispatcher,
                ops...>::template run<int, config, globals,
                                        ::megakernel::state<config>>(opcode,
                                                                    g, kvms);

            // broadcast the result to all lanes
            num_semaphores[kvms.instruction_ring] = __shfl_sync(
                0xffffffff, num_semaphores[kvms.instruction_ring], 0);
        }

        if (laneid == 0) {
            kvms.internal_record(detail::TIMING_EVENT_SPECIAL_CONTROLLER_READY);
            // Step 4. Let the rest of the world know that next instruction is
            // ready to roll!
            arrive(kvms.instruction_arrived[kvms.instruction_ring], 1);
        }

        // Save the global instruction index for work stealing
        if constexpr (config::ENABLE_GLOBAL_WORK_QUEUE) {
            #pragma unroll
            for(int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES-1; i++) {
                last_global_instruction_indices[i] = last_global_instruction_indices[i+1];
            }
            last_global_instruction_indices[config::INSTRUCTION_PIPELINE_STAGES-1] = global_instruction_index;
        }

    }

    // invalidate remaining semaphores and write out remaining timings
    int true_num_iters = kvms.instruction_index;
    for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES; i++) {

        auto instruction_index = true_num_iters - config::INSTRUCTION_PIPELINE_STAGES + i;

        int true_index; 
        if constexpr (config::ENABLE_GLOBAL_WORK_QUEUE) {
            #pragma unroll
            for(int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES-1; i++) {
                last_global_instruction_indices[i] = last_global_instruction_indices[i+1];
            }
            true_index = last_global_instruction_indices[0];
        } else {
            true_index = instruction_index;
        }

        if (instruction_index < 0) {
            continue;
        }

        auto instruction_ring =
            instruction_index % config::INSTRUCTION_PIPELINE_STAGES;

        auto phasebit = (instruction_index / config::INSTRUCTION_PIPELINE_STAGES) & 1;
        kvms.instruction_index = instruction_index;
        kvms.instruction_ring = instruction_ring;

        kittens::wait(kvms.instruction_finished[instruction_ring], phasebit);


        if constexpr (config::TIMING_RECORD_ENABLED) {
            if (laneid == 0) kvms.internal_record(detail::TIMING_EVENT_SPECIAL_CONTROLLER_CLEANUP);
        }

        // don't need to invalidate on teardown
        int num_to_invalidate = num_semaphores[instruction_ring];
        for (int sem_idx = laneid; sem_idx < num_to_invalidate; sem_idx += 32) {
            invalidate_semaphore(
                kvms.all_instructions[instruction_ring]
                    .semaphores[sem_idx]);
        }

        if constexpr (config::TIMING_RECORD_ENABLED) {
            store_timings_and_reset<config, globals>(
                &kvms.all_instructions[instruction_ring].timings[0],
                true_index, g);
        }

    }

}

} // namespace controller
} // namespace megakernel
