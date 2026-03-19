#pragma once

#include "kittens.cuh"

namespace megakittens {

struct default_config {
    static constexpr int INSTRUCTION_PIPE_STAGES = 2; // should not change
    static constexpr int INSTRUCTION_WIDTH = 32;      // 128 bytes per instruction
    using instruction_t = int[INSTRUCTION_WIDTH];
    static_assert(INSTRUCTION_WIDTH <= 32); // for warp parallel processing

    static constexpr int DYNAMIC_SEMAPHORES = 32;
    static_assert(DYNAMIC_SEMAPHORES <= 32); // for warp parallel processing

    static constexpr int NUM_CONSUMER_WARPS = 16;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int CLUSTER_BLOCKS = 1;
    static constexpr int MAX_SHARED_MEMORY = ::kittens::MAX_SHARED_MEMORY;

    // Shared memory declared statically
    static constexpr int SCRATCH_BYTES = 4096;
    static constexpr int STATIC_SHARED_MEMORY =
        512 + INSTRUCTION_PIPELINE_STAGES *
                  (SCRATCH_BYTES + (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 +
                   DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY = kittens::MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    // Shared memory declared dynamically
    static constexpr int PAGE_SIZE = 16384;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES == 13, "NUM_PAGES must be 13");
    static_assert(NUM_PAGES <= 32); // for warp parallel processing

    static constexpr bool TIMING_RECORD_ENABLED = false;

    static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS = 20;

    static constexpr int CONSUMER_REGISTERS = 104;
    static constexpr int NON_CONSUMER_REGISTERS = 64;
};

template <typename Config>
using instruction_layout = kittens::gl<int, 1, 1, -1, Config::INSTRUCTION_WIDTH>;

template <typename Config> 
struct __align__(128) instruction_state_t {
    Config::instruction_t instruction;
    int pid_order[Config::NUM_PAGES];
    int _padding[((Config::NUM_PAGES + 31) & ~31) - Config::NUM_PAGES]; // round up to multiple of 32
    kittens::semaphore semaphores[Config::DYNAMIC_SEMAPHORES];
    int scratch[Config::SCRATCH_BYTES / 4]; // todo keep it really /4?
};

template <typename Config>
struct page_t {
    int data[Config::PAGE_SIZE / sizeof(int)];
    __device__ inline void *ptr(int byte_offset = 0) {
        return (void *)(data + byte_offset / sizeof(int));
    }
    __device__ inline const void *ptr(int byte_offset = 0) const {
        return (const void *)(data + byte_offset / sizeof(int));
    }
};

template <typename Config>
struct state_t {
    uint32_t iter;
    uint32_t stage;

    kittens::clc::handle (&clc_handle)[Config::INSTRUCTION_PIPE_STAGES];
    kittens::semaphore (&clc_arrived)[Config::INSTRUCTION_PIPE_STAGES];

    instruction_state_t<Config> (&instruction_states)[Config::INSTRUCTION_PIPE_STAGES];
    kittens::semaphore (&instruction_arrived)[Config::INSTRUCTION_PIPE_STAGES];
    kittens::semaphore (&instruction_finished)[Config::INSTRUCTION_PIPE_STAGES];

    page_t<Config> (&pages)[Config::NUM_PAGES];
    kittens::semaphore (&page_finished)[Config::NUM_PAGES];

    kittens::semaphore &tensor_finished;
    kittens::tensor_allocator<1, 2> &tensor_alloc;

    __device__ inline int (&instruction())[Config::INSTRUCTION_WIDTH] {
        return instruction_states[stage].instruction;
    }
    __device__ inline const int (&instruction() const)[Config::INSTRUCTION_WIDTH] {
        return instruction_states[stage].instruction;
    }
    __device__ inline int (&pid_order())[Config::NUM_PAGES] {
        return instruction_states[stage].pid_order;
    }
    __device__ inline const int (&pid_order() const)[Config::NUM_PAGES] {
        return instruction_states[stage].pid_order;
    }
    __device__ inline kittens::semaphore (&semaphores())[Config::DYNAMIC_SEMAPHORES] {
        return instruction_states[stage].semaphores;
    }
    __device__ inline const kittens::semaphore (&semaphores() const)[Config::DYNAMIC_SEMAPHORES] {
        return instruction_states[stage].semaphores;
    }
    __device__ inline void *scratch() const {
        return (void *)&instruction_states[stage].scratch[0];
    }

    __device__ inline int pid(int lid) {
        return pid_order()[lid];
    }
    __device__ inline void finish_page(int pid, int count) {
        arrive(page_finished[pid], count);
    }
    __device__ inline void warp_finish_page(int pid, int count) {
        if (kittens::warp::laneid() == 0)
            finish_page(pid, count);
    }
    __device__ inline void wait_page_ready(int pid) {
        kittens::wait(page_finished[pid], iter & 0b1);
    }
    __device__ inline void wait_tensor_ready() {
        kittens::wait(tensor_finished, iter & 0b1);
    }
    __device__ inline void wait_semaphores_ready() {
        kittens::wait(semaphores_ready, iter & 0b1);
    }
};

} // namespace megakittens
