#pragma once

#include "kittens.cuh"

#include "schema.cuh"
#include "utils.cuh"
#include "controller.cuh"
#include "workers.cuh"

namespace megakittens {

template <typename Config, typename Globals>
__global__ __launch_bounds__(Config::NUM_THREADS, Config::MIN_BLOCKS_PER_SM)
void kernel(const __grid_constant__ Globals g) {
    // Allocate shared memory
    __shared__ alignas(128) instruction_state_t<Config> instruction_states[Config::INSTRUCTION_PIPE_STAGES];
    __shared__ kittens::clc::handle clc_handle[Config::INSTRUCTION_PIPE_STAGES];
    __shared__ kittens::semaphore clc_arrived[Config::INSTRUCTION_PIPE_STAGES];
    __shared__ kittens::semaphore instruction_arrived[Config::INSTRUCTION_PIPE_STAGES];
    __shared__ kittens::semaphore instruction_finished[Config::INSTRUCTION_PIPE_STAGES];
    __shared__ kittens::semaphore page_finished[Config::NUM_PAGES];
    __shared__ kittens::semaphore tensor_finished;
    extern __shared__ int __shm[];
    page_t<Config> (&pages)[Config::NUM_PAGES] = *reinterpret_cast<page_t<Config>(*)[Config::NUM_PAGES]>(
            reinterpret_cast<void *>(((uint64_t)&__shm[0] + 1023) & ~(uint64_t)1023));

    // Allocate tensor memory
    kittens::tensor_allocator<1, Config::CLUSTER_SIZE> tensor_alloc;

    // Instantiate MegaKittens state
    state_t<Config> s{0, 0, clc_handle, clc_arrived,
                      instruction_states, instruction_arrived, instruction_finished,
                      pages, page_finished, tensor_finished, tensor_alloc,
                      g.timings_ptr, g.timings_stride, clock64()};

    // Initialize common semaphores
    if (threadIdx.x < Config::INSTRUCTION_PIPE_STAGES) {
        init_semaphore(instruction_arrived[threadIdx.x], Config::CLUSTER_SIZE);
    } else if (threadIdx.x < Config::INSTRUCTION_PIPE_STAGES*2) {
        init_semaphore(instruction_finished[threadIdx.x - Config::INSTRUCTION_PIPE_STAGES], Config::NUM_WARPS - 1);
    } else if (threadIdx.x < Config::INSTRUCTION_PIPE_STAGES*3) {
        init_semaphore(clc_arrived[threadIdx.x - Config::INSTRUCTION_PIPE_STAGES*2], 1);
    } else if (threadIdx.x < Config::INSTRUCTION_PIPE_STAGES*3 + Config::NUM_PAGES) {
        init_semaphore(page_finished[threadIdx.x - Config::INSTRUCTION_PIPE_STAGES*3], 1);
        arrive(page_finished[threadIdx.x - Config::INSTRUCTION_PIPE_STAGES*3], 1);
    } else if (threadIdx.x < Config::INSTRUCTION_PIPE_STAGES*3 + Config::NUM_PAGES + 1) {
        init_semaphore(tensor_finished, 1);
        arrive(tensor_finished, 1);
    }
    kittens::everyone::tma::cluster::sync();
    kittens::pdl::wait();

    // Initiate the main loops
    if (kittens::warpid() < Config::NUM_CONSUMER_WARPS) {
        kittens::warpgroup::increase_registers<Config::CONSUMER_REGISTERS>();
        consumer_loop<Config, Globals>(g, s);
    } else {
        kittens::warpgroup::decrease_registers<Config::NON_CONSUMER_REGISTERS>();
        switch (kittens::warpgroup::warpid()) {
            case 0:
                controller_loop<Config, Globals>(g, s);
                break;
            case 1:
                loader_loop<Config, Globals>(g, s);
                break;
            case 2:
                launcher_loop<Config, Globals>(g, s);
                break;
            case 3:
                storer_loop<Config, Globals>(g, s);
                break;
            default:
                asm volatile("{trap;\n}");
        }
    }

    // Sync all threads in the cluster before exiting
    kittens::everyone::tma::cluster::sync();
}

} // namespace megakittens
