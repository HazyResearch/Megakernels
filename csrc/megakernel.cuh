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
__device__ __forceinline__ void mk_internal(const globals &g) {
    uint64_t start_time = timestamp();
#ifdef MK_DEBUG
    if (threadIdx.x == 0)
        printf("Thread %d: Kernel launched\n", threadIdx.x);
    group<config::NUM_WARPS>::sync(15);
#endif
    __shared__ alignas(128) instruction_state_t<config>
        instruction_state[config::INSTRUCTION_PIPELINE_STAGES];
    __shared__ kittens::semaphore
        page_finished[config::NUM_PAGES]
                     [config::INSTRUCTION_PIPELINE_STAGES_BITS],
        instruction_arrived[config::INSTRUCTION_PIPELINE_STAGES],
        instruction_finished[config::INSTRUCTION_PIPELINE_STAGES],
        instruction_fetch_ready,
#ifdef KITTENS_BLACKWELL
        tensor_finished,
#endif
        semaphores_ready;
    extern __shared__ int __shm[];
    void *aligned_shm_addr =
        (void *)((1023 + (uint64_t)&__shm[0]) & ~(uint64_t)1023);
    typename state<config>::page_array_t &pages =
        *reinterpret_cast<typename state<config>::page_array_t *>(
            aligned_shm_addr);
#ifdef KITTENS_BLACKWELL
    typename state<config>::tensor_allocator_t tensor_alloc{};
#endif

#ifdef MK_DEBUG
    if (threadIdx.x == 0)
        printf("Thread %d: Pre-MKS creation\n", threadIdx.x);
    group<config::NUM_WARPS>::sync(15);
#endif
    state<config> mks{instruction_state,
                      instruction_arrived,
                      instruction_finished,
                      0,
                      0,
                      instruction_fetch_ready,
                      pages,
                      page_finished,
#ifdef KITTENS_BLACKWELL
                      tensor_finished,
#endif
                      semaphores_ready,
                      start_time,
                      0 // write_event
#ifdef KITTENS_BLACKWELL
                      ,
                      tensor_alloc
#endif
    }; // megakernel state

#ifdef MK_DEBUG
    if (threadIdx.x == 0)
        printf("Thread %d: Created MKS\n", threadIdx.x);
    group<config::NUM_WARPS>::sync(15);
#endif

    // Zero initial timings memory.
    if (threadIdx.x < config::TIMING_WIDTH) {
#pragma unroll
        for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES; i++) {
            instruction_state[i].timings[threadIdx.x] = 0;
        }
    }

    if(threadIdx.x == 0) init_semaphore(instruction_fetch_ready, config::NUM_CONSUMER_WARPS);
    if (threadIdx.x < config::INSTRUCTION_PIPELINE_STAGES) {
        init_semaphore(instruction_arrived[threadIdx.x], 1);
        init_semaphore(instruction_finished[threadIdx.x],
                       config::NUM_WARPS - 1);
    }
    if (threadIdx.x < config::NUM_PAGES) {
        for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES_BITS; i++) {
            auto count = config::NUM_CONSUMER_WARPS * (1 << i);
            init_semaphore(page_finished[threadIdx.x][i], count);
            arrive(page_finished[threadIdx.x][i], count);
        }
    }
    if (threadIdx.x == 0) {
#ifdef KITTENS_BLACKWELL
        init_semaphore(tensor_finished, config::NUM_CONSUMER_WARPS);
        arrive(tensor_finished,
               config::NUM_CONSUMER_WARPS); // Flip to state 0, to mark that it
                                            // starts as available.
#endif
        init_semaphore(semaphores_ready, 1);
    }

    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    if (config::CLUSTER_BLOCKS == 1)
        kittens::everyone::sync(15); // all warps must arrive here, confirming semaphore
                            // initialization is visible to all threads.
    else
        kittens::everyone::tma::cluster::sync();

#ifdef MK_DEBUG
    if (blockIdx.x == 0 && threadIdx.x == 0)
        mks.print();
#endif

    if (kittens::warpid() < config::NUM_CONSUMER_WARPS) {
        kittens::warpgroup::increase_registers<config::CONSUMER_REGISTERS>();
        ::megakernel::consumer::main_loop<config, globals, ops...>(g, mks);
    } else {
        kittens::warpgroup::decrease_registers<config::NON_CONSUMER_REGISTERS>();
        switch (kittens::warpgroup::warpid()) {
        case 0:
            ::megakernel::loader::main_loop<config, globals, ops...>(g, mks);
            break;
        case 1:
            ::megakernel::storer::main_loop<config, globals, ops...>(g, mks);
            break;
        case 2:
            ::megakernel::launcher::main_loop<config, globals, ops...>(g, mks);
            break;
        case 3:
            ::megakernel::controller::main_loop<config, globals, ops...>(g,
                                                                         mks);
            break;
        default:
            asm volatile("trap;");
        }
    }

#ifdef MK_DEBUG
    printf("Thread %d arriving at final barrier\n", threadIdx.x);
#endif

    if (config::CLUSTER_BLOCKS > 1)
        kittens::everyone::tma::cluster::sync();
    else
        kittens::everyone::sync(15);

#ifdef MK_DEBUG
    uint64_t end_time = (uint64_t)clock64();
    if (threadIdx.x == 0)
        printf("Overall VM execution time: %lu\n", end_time - start_time);
#endif
}

// Forward a NoOp to the VM, to ensure that the VM can support zeros.
template <typename config, typename globals, typename... ops>
struct megakernel_wrapper {
    __device__ __forceinline__ static void run(const globals &g) {
        mk_internal<config, globals, NoOp<config>, ops...>(g);
    }
};

#ifdef CUTLASS_NAME
// Add the name "cutlass" to the kernel so that ptxas compiles it to run faster. (No, I'm not kidding.)
template <typename config, typename globals, typename... ops>
__launch_bounds__(config::NUM_THREADS, 1)
__cluster_dims__(config::CLUSTER_BLOCKS)
__global__ void cutlass_mk(const __grid_constant__ globals g) {
    megakernel_wrapper<config, globals, ops...>::run(g);
}
#define mk cutlass_mk
#else
template <typename config, typename globals, typename... ops>
__launch_bounds__(config::NUM_THREADS, 1)
__cluster_dims__(config::CLUSTER_BLOCKS)
__global__ void mk(const __grid_constant__ globals g) {
    megakernel_wrapper<config, globals, ops...>::run(g);
}
#endif

} // namespace megakernel