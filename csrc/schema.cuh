#pragma once

#include "kittens.cuh"

namespace megakittens {

struct default_config {
    // Instruction pipeline stages (should NOT change)
    static constexpr int INSTRUCTION_PIPE_STAGES = 2;

    static constexpr int INSTRUCTION_WIDTH = 32; // 128 bytes per instruction.
    using instruction_t = int[INSTRUCTION_WIDTH];

    // Timing info
    static constexpr int TIMING_WIDTH = 128;
    using timing_t = int[TIMING_WIDTH];

    // How many semaphores are available for dynamic use?
    static constexpr int DYNAMIC_SEMAPHORES = 32;

    // One controller warp, one load warp, one store warp, and one mma warp.
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
    static constexpr int DYNAMIC_SHARED_MEMORY =
        ::kittens::MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    // Shared memory declared dynamically
    static constexpr int PAGE_SIZE = 16384;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES == 13, "NUM_PAGES must be 13");

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
    int _padding[((Config::NUM_PAGES + 31) & ~31) - Config::NUM_PAGES]; // Round up to multiple of 32
    kittens::semaphore semaphores[Config::DYNAMIC_SEMAPHORES];
    int scratch[Config::SCRATCH_BYTES / 4];
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
    instruction_state_t<Config> (&instruction_states)[Config::INSTRUCTION_PIPE_STAGES];

    kittens::semaphore (&instruction_arrived)[Config::INSTRUCTION_PIPE_STAGES];
    kittens::semaphore (&instruction_finished)[Config::INSTRUCTION_PIPE_STAGES];
    kittens::semaphore &instruction_fetch_ready;

    kittens::semaphore &semaphores_ready;

    page_t<Config> (&pages)[Config::NUM_PAGES];
    kittens::semaphore (&page_finished)[Config::NUM_PAGES][Config::INSTRUCTION_PIPE_STAGES_BITS];

    int instruction_index;
    int instruction_ring;

    kittens::semaphore &tensor_finished;
    kittens::tensor_allocator<1, 2> &tensor_alloc;

    uint32_t pid_order_shared_addr;

    __device__ inline int (&instruction())[Config::INSTRUCTION_WIDTH] {
        return instruction_states[instruction_ring].instruction;
    }
    __device__ inline const int (&instruction() const)[Config::INSTRUCTION_WIDTH] {
        return instruction_states[instruction_ring].instruction;
    }
    __device__ inline int (&pid_order())[Config::NUM_PAGES] {
        return instruction_states[instruction_ring].pid_order;
    }
    __device__ inline const int (&pid_order() const)[Config::NUM_PAGES] {
        return instruction_states[instruction_ring].pid_order;
    }
    __device__ inline void *scratch() const {
        return (void *)&instruction_states[instruction_ring].scratch[0];
    }
    __device__ inline kittens::semaphore (&semaphores())[Config::DYNAMIC_SEMAPHORES] {
        return instruction_states[instruction_ring].semaphores;
    }
    __device__ inline const kittens::semaphore (&semaphores() const)[Config::DYNAMIC_SEMAPHORES] {
        return instruction_states[instruction_ring].semaphores;
    }

    __device__ inline void await_instruction() {
        kittens::wait(instruction_arrived[instruction_ring], (instruction_index / Config::INSTRUCTION_PIPE_STAGES) & 0b1);
        pid_order_shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&(pid_order()[0])));
    }
    __device__ inline void next_instruction() {
        if (kittens::laneid() == 0) // TODO: replace all of these with elect sync
            kittens::arrive(instruction_finished[instruction_ring]);
        instruction_index++;
        instruction_ring = kittens::ring_advance<Config::INSTRUCTION_PIPE_STAGES>(instruction_ring);
    }

    __device__ inline int pid(int lid) {
        int ret;
        kittens::move<int>::lds(ret, pid_order_shared_addr + lid * sizeof(int));
        return ret;
    }
    __device__ inline void finish_page(int pid, int count) {
        arrive(page_finished[pid], count);
    }
    __device__ inline void warp_finish_page(int pid, int count) {
        if (kittens::warp::laneid() == 0)
            finish_page(pid, count);
    }
    __device__ inline void wait_page_ready(int pid) {
        kittens::wait(page_finished[pid], instruction_index & 0b1);
    }
    __device__ inline void wait_tensor_ready() {
        kittens::wait(tensor_finished, instruction_index & 0b1);
    }
    __device__ inline void wait_semaphores_ready() {
        kittens::wait(semaphores_ready, instruction_index & 0b1);
    }
};

} // namespace megakittens
