#pragma once

#include "kittens.cuh"

#include "schema.cuh"
// #include "controller.cuh"
// #include "workers.cuh"
// #include "ops/ops.cuh"

namespace megakittens {

template <typename Config, typename Globals, typename ...Ops>
__global__ __launch_bounds__(Config::NUM_THREADS, Config::MIN_BLOCKS_PER_SM)
void kernel(const __grid_constant__ Globals g) {

    using C = Config;
    using G = Globals;

    // Allocate shared memory
    __shared__ alignas(128) instruction_state_t<Config> instruction_states[Config::INSTRUCTION_PIPE_STAGES];
    __shared__ kittens::clc::handle clc_handle[Config::INSTRUCTION_PIPE_STAGES];
    __shared__ kittens::semaphore clc_arrived[Config::INSTRUCTION_PIPE_STAGES];
    __shared__ kittens::semaphore instruction_arrived[Config::INSTRUCTION_PIPE_STAGES];
    __shared__ kittens::semaphore instruction_finished[Config::INSTRUCTION_PIPE_STAGES];
    __shared__ kittens::semaphore page_finished[Config::NUM_PAGES];
    __shared__ kittens::semaphore tensor_finished;
    extern __shared__ int __shm[];
    page_t<Config> (&pages)[Config::NUM_PAGES] = *reinterpret_cast<page_t<Config> (*)[Config::NUM_PAGES]>(
            reinterpret_cast<void *>(((uint64_t)&__shm[0] + 1023) & ~(uint64_t)1023));

    // Allocate tensor memory
    typename state<Config>::tensor_allocator_t tensor_alloc;

    // Instantiate MegaKittens state
    state<Config> s{0, 0, clc_handle, clc_arrived, instruction_states, 
                    instruction_arrived, instruction_finished,
                    pages, page_finished,
                    tensor_finished, tensor_alloc};

    // Initialize all semaphores
    if (threadIdx.x == 0) {
        init_semaphore(tensor_finished, Config::NUM_CONSUMER_WARPS);
        arrive(tensor_finished, Config::NUM_CONSUMER_WARPS); // flip the phasebit
    }
    if (threadIdx.x < Config::INSTRUCTION_PIPE_STAGES) {
        init_semaphore(instruction_arrived[threadIdx.x], 1);
        init_semaphore(instruction_finished[threadIdx.x], Config::NUM_WARPS - 1);
    }
    if (threadIdx.x < Config::NUM_PAGES) {
        init_semaphore(page_finished[threadIdx.x], Config::NUM_CONSUMER_WARPS);
        arrive(page_finished[threadIdx.x], Config::NUM_CONSUMER_WARPS); // flip the phasebit
    }
    kittens::everyone::tma::cluster::sync();

    // Initiate the main loops
    if (kittens::warpid() < Config::NUM_CONSUMER_WARPS) {
        kittens::warpgroup::increase_registers<Config::CONSUMER_REGISTERS>();
        consumer::main_loop<Config, Globals, Ops...>(g, s);
    } else {
        kittens::warpgroup::decrease_registers<Config::NON_CONSUMER_REGISTERS>();
        switch (kittens::warpgroup::warpid()) {
            case 0:
                controller_loop<Config, Globals, Ops...>(g, s);
                break;
            case 1:
                loader::main_loop<Config, Globals, Ops...>(g, s);
                break;
            case 2:
                launcher::main_loop<Config, Globals, Ops...>(g, s);
                break;
            case 3:
                storer::main_loop<Config, Globals, Ops...>(g, s);
                break;
            default:
                asm volatile("{trap;\n}");
        }
    }

    // Sync all threads in the cluster before exiting
    kittens::everyone::tma::cluster::sync();
}

} // namespace megakittens
