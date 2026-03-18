#pragma once

#include "kittens.cuh"
#include "config.cuh"
#include "util.cuh"
#include "controller/controller.cuh"
#include "launcher.cuh"
#include "storer.cuh"
#include "loader.cuh"
#include "consumer.cuh"
#include "noop.cuh"

namespace megakernel {

template <typename config, typename globals, typename... ops>
__device__ __forceinline__ void _megakernel(const globals &g) {
    // Allocate shared memory
    __shared__ alignas(128) instruction_state_t<config> instruction_states[config::INSTRUCTION_PIPE_STAGES];
    __shared__ kittens::semaphore page_finished[config::NUM_PAGES];
    __shared__ kittens::semaphore instruction_arrived[config::INSTRUCTION_PIPE_STAGES];
    __shared__ kittens::semaphore instruction_finished[config::INSTRUCTION_PIPE_STAGES];
    __shared__ kittens::semaphore instruction_fetch_ready;
    __shared__ kittens::semaphore tensor_finished;
    __shared__ kittens::semaphore semaphores_ready;
    extern __shared__ int __shm[];
    page_t<config> (&pages)[config::NUM_PAGES] = *reinterpret_cast<page_t<config> (*)[config::NUM_PAGES]>(
            reinterpret_cast<void *>(((uint64_t)&__shm[0] + 1023) & ~(uint64_t)1023));

    // Allocate tensor memory
    typename state<config>::tensor_allocator_t tensor_alloc;

    // Instantiate MegaKernel state
    state<config> megakernel_state{instruction_states, instruction_arrived, instruction_finished,
                                   instruction_fetch_ready, semaphores_ready, pages, page_finished,
                                   0, 0, tensor_finished, tensor_alloc, 0};

    // Initialize all semaphores
    if (threadIdx.x == 0) {
        init_semaphore(instruction_fetch_ready, config::NUM_CONSUMER_WARPS);
        init_semaphore(tensor_finished, config::NUM_CONSUMER_WARPS);
        arrive(tensor_finished, config::NUM_CONSUMER_WARPS); // flip the phasebit
        init_semaphore(semaphores_ready, 1);
    }
    if (threadIdx.x < config::INSTRUCTION_PIPE_STAGES) {
        init_semaphore(instruction_arrived[threadIdx.x], 1);
        init_semaphore(instruction_finished[threadIdx.x], config::NUM_WARPS - 1);
    }
    if (threadIdx.x < config::NUM_PAGES) {
        init_semaphore(page_finished[threadIdx.x], config::NUM_CONSUMER_WARPS);
        arrive(page_finished[threadIdx.x], config::NUM_CONSUMER_WARPS); // flip the phasebit
    }
    kittens::everyone::tma::cluster::sync();

    // Initiate the main loops
    if (kittens::warpid() < config::NUM_CONSUMER_WARPS) {
        kittens::warpgroup::increase_registers<config::CONSUMER_REGISTERS>();
        consumer::main_loop<config, globals, ops...>(g, megakernel_state);
    } else {
        kittens::warpgroup::decrease_registers<config::NON_CONSUMER_REGISTERS>();
        switch (kittens::warpgroup::warpid()) {
        case 0:
            loader::main_loop<config, globals, ops...>(g, megakernel_state);
            break;
        case 1:
            storer::main_loop<config, globals, ops...>(g, megakernel_state);
            break;
        case 2:
            launcher::main_loop<config, globals, ops...>(g, megakernel_state);
            break;
        case 3:
            controller::main_loop<config, globals, ops...>(g, megakernel_state);
            break;
        default:
            asm volatile("{trap;\n}");
        }
    }

    // Sync all threads in the cluster before exiting
    kittens::everyone::tma::cluster::sync();
}

template <typename config, typename globals, typename... ops>
__device__ __forceinline__ void megakernel(const __grid_constant__ globals g) {
    // _megakernel<config, globals, NoOp<config>, ops...>(g);
}

} // namespace megakernel
