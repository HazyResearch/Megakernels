#pragma once

#include "kittens.cuh"

namespace megakittens {

struct default_config {
    static constexpr int INSTRUCTION_PIPE_STAGES = 2; // should not change
    static constexpr int INSTRUCTION_WIDTH = 32; // 128 bytes per instruction
    using instruction_t = int[INSTRUCTION_WIDTH];
    static_assert(INSTRUCTION_WIDTH <= 32); // for warp parallel processing

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int MIN_BLOCKS_PER_SM = 1;

    static constexpr int NUM_CONSUMER_WARPS = 8;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
    static constexpr int CONSUMER_REGISTERS = 104;
    static constexpr int NON_CONSUMER_REGISTERS = 64;

    static constexpr int DYNAMIC_SEMAPHORES = 32;
    static_assert(DYNAMIC_SEMAPHORES <= 32); // for warp parallel processing

    static constexpr int SCRATCH_BYTES = 4096;
    static constexpr int STATIC_SHARED_MEMORY = 256 + INSTRUCTION_PIPE_STAGES *
        (INSTRUCTION_WIDTH*4 + 128 + DYNAMIC_SEMAPHORES*8 + SCRATCH_BYTES); // sizeof(instruction_state_t)
    static constexpr int DYNAMIC_SHARED_MEMORY = kittens::MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int PAGE_SIZE = 16384;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES <= 32); // for warp parallel processing

    static constexpr bool SPIN_LOOP_SLEEP_NS = 20;
};

template <typename Config>
struct __align__(128) instruction_state_t {
    Config::instruction_t instruction;
    int pid_order[Config::NUM_PAGES];
    int _padding[((Config::NUM_PAGES + 31) & ~31) - Config::NUM_PAGES]; // make pid_order + _padding 128 bytes
    kittens::semaphore semaphores[Config::DYNAMIC_SEMAPHORES];
    int scratch[Config::SCRATCH_BYTES/sizeof(int)];
};

template <typename Config>
struct page_t {
    int data[Config::PAGE_SIZE/sizeof(int)];
    __device__ __forceinline__ void *ptr(int byte_offset = 0) {
        return reinterpret_cast<void *>(reinterpret_cast<uint64_t>(&data[0])+byte_offset);
    }
    __device__ __forceinline__ const void *ptr(int byte_offset = 0) const {
        return reinterpret_cast<const void *>(reinterpret_cast<uint64_t>(&data[0])+byte_offset);
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

    __device__ __forceinline__ int (&instruction())[Config::INSTRUCTION_WIDTH] {
        return instruction_states[stage].instruction;
    }
    __device__ __forceinline__ const int (&instruction() const)[Config::INSTRUCTION_WIDTH] {
        return instruction_states[stage].instruction;
    }
    __device__ __forceinline__ int (&pid_order())[Config::NUM_PAGES] {
        return instruction_states[stage].pid_order;
    }
    __device__ __forceinline__ const int (&pid_order() const)[Config::NUM_PAGES] {
        return instruction_states[stage].pid_order;
    }
    __device__ __forceinline__ kittens::semaphore (&semaphores())[Config::DYNAMIC_SEMAPHORES] {
        return instruction_states[stage].semaphores;
    }
    __device__ __forceinline__ const kittens::semaphore (&semaphores() const)[Config::DYNAMIC_SEMAPHORES] {
        return instruction_states[stage].semaphores;
    }
    __device__ __forceinline__ void *scratch() const {
        return reinterpret_cast<void *>(&instruction_states[stage].scratch[0]);
    }
    __device__ __forceinline__ int pid(int lid) {
        return pid_order()[lid];
    }
    __device__ __forceinline__ void page_wait(int pid) {
        kittens::wait(page_finished[pid], iter&0b1);
    }
    __device__ __forceinline__ void page_finish(int pid, int count) {
        kittens::arrive(page_finished[pid], count);
    }
    __device__ __forceinline__ void tensor_wait() {
        kittens::wait(tensor_finished, iter&0b1);
    }
    __device__ __forceinline__ void tensor_finish(int count) {
        kittens::arrive(tensor_finished, count);
    }
};

} // namespace megakittens
