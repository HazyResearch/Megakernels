#pragma once

#include "kittens.cuh"

namespace megakittens {
namespace controller {

template <typename config, typename globals>
__device__ inline bool load_instructions(int *instruction,
                                         int instruction_index,
                                         const globals &g) {
    static_assert(config::INSTRUCTION_WIDTH <= 32);
    auto laneid = ::kittens::laneid();
    int *src_ptr;
    if constexpr (kittens::ducks::gl::all<decltype(g.instructions)>) {
        if constexpr (config::ENABLE_GLOBAL_WORK_QUEUE) {
            src_ptr = &g.instructions[kittens::coord<>{instruction_index, 0}];
        }
        else {
            src_ptr = &g.instructions[kittens::coord<>{(int)(get_worker_id()), instruction_index, 0}];
        }
        static_assert(std::is_same<decltype(src_ptr), int *>::value, "src_ptr is not an int*");
        if (laneid < config::INSTRUCTION_WIDTH)
            instruction[laneid] = src_ptr[laneid];
    } else if constexpr (kittens::ducks::pgl::all<decltype(g.instructions)>) {
        if constexpr (config::ENABLE_GLOBAL_WORK_QUEUE) {
            src_ptr = &g.instructions[g.dev_idx][kittens::coord<>{instruction_index, 0}];
        }
        else {
            src_ptr = &g.instructions[g.dev_idx][kittens::coord<>{(int)(get_worker_id()), instruction_index, 0}];
        }
        static_assert(std::is_same<decltype(src_ptr), int *>::value, "src_ptr is not an int*");
        if (laneid < config::INSTRUCTION_WIDTH)
            instruction[laneid] = src_ptr[laneid];
    }
    kittens::warp::sync();
    return instruction[0] != -1;
}


template <typename config, typename globals>
struct page_allocator_op_dispatcher {
    template <typename op> struct dispatcher {
        __device__ static inline int
        run(const globals &g, typename config::instruction_t &instruction,
            int &query) {
            return op::controller::release_lid(g, instruction, query);
        }
    };
};

template <typename config, typename globals, typename... ops>
__device__ void inline page_allocator_loop(const globals &g,
                                           ::megakittens::state<config> &kvms) {
    static_assert(config::INSTRUCTION_PIPELINE_STAGES <= 16,
                  "This would be an absurd thing to do.");
    constexpr uint32_t membermask = 0xFFFFFFFF >> (32 - config::NUM_PAGES);
    int num_iters = g.instructions.rows();
    for (kvms.instruction_index = 0, kvms.instruction_ring = 0;
         kvms.instruction_index < num_iters;
         kvms.instruction_index++,
        kvms.instruction_ring =
             ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(
                 kvms.instruction_ring)) {

        int phasebit =
            (kvms.instruction_index / config::INSTRUCTION_PIPELINE_STAGES - 1) &
            1;
        if (kvms.instruction_index >= config::INSTRUCTION_PIPELINE_STAGES)
            kittens::wait(kvms.instruction_finished[kvms.instruction_ring], phasebit);

        int next_pid;
        if (kvms.instruction_index == 0)
            next_pid = kittens::laneid();
        else {
            int last_instruction_ring =
                (kvms.instruction_ring + config::INSTRUCTION_PIPELINE_STAGES -
                 1) %
                config::INSTRUCTION_PIPELINE_STAGES;
            kittens::wait(kvms.instruction_arrived[last_instruction_ring],
                 ((kvms.instruction_index - 1) /
                  config::INSTRUCTION_PIPELINE_STAGES) &
                     1);
            int lane = kittens::laneid();
            int opcode =
                kvms.all_instructions[last_instruction_ring].instructions[0];
            int lid = dispatch_op<
                page_allocator_op_dispatcher<config, globals>::dispatcher,
                ops...>::template run<int, config, globals,
                                      config::instruction_t, int>(
                opcode, g,
                kvms.all_instructions[last_instruction_ring].instructions,
                lane);
            next_pid =
                kvms.all_instructions[last_instruction_ring].pid_order[lid];
        }
        kvms.pid_order()[kittens::laneid()] = next_pid;
        asm volatile("bar.warp.sync %0;\n" ::"n"(membermask));
        if (kittens::laneid() == 0)
            kittens::arrive(kvms.instruction_arrived[kvms.instruction_ring], 1);
    }
}

template <typename config, typename globals>
struct semaphore_constructor_op_dispatcher {
    template <typename op> struct dispatcher {
        __device__ static inline int
        run(const globals &g, ::megakittens::state<config> &kvms) {
            auto out = op::controller::init_semaphores(g, kvms);
            asm volatile("{fence.proxy.async.shared::cta;\n}" ::: "memory");
            return out;
        }
    };
};

template <typename config, typename globals, typename... ops>
__device__ void inline semaphore_constructor_loop(
    const globals &g, ::megakittens::state<config> &kvms) {
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
                                      ::megakittens::state<config>>(
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

template <typename config, typename globals, typename... ops>
__device__ void main_loop(const globals &g, ::megakittens::state<config> &kvms) {
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
                                        ::megakittens::state<config>>(opcode,
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
} // namespace megakittens
