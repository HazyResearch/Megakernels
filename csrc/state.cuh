#pragma once

#include "kittens.cuh"
#include "config.cuh"

namespace megakernel {

template <typename config> 
struct __align__(128) instruction_state_t {
    config::instruction_t instruction;
    int pid_order[config::NUM_PAGES];
    int _padding[((config::NUM_PAGES + 31) & ~31) - config::NUM_PAGES]; // Round up to multiple of 32
    kittens::semaphore semaphores[config::DYNAMIC_SEMAPHORES];
    int scratch[config::SCRATCH_BYTES / 4];
};

template <typename config>
struct page_t {
    int data[config::PAGE_SIZE / sizeof(int)];
    __device__ inline void *ptr(int byte_offset = 0) {
        return (void *)(data + byte_offset / sizeof(int));
    }
    __device__ inline const void *ptr(int byte_offset = 0) const {
        return (const void *)(data + byte_offset / sizeof(int));
    }
};

template <typename config>
struct state_t {
    instruction_state_t<config> (&instruction_states)[config::INSTRUCTION_PIPE_STAGES];

    kittens::semaphore (&instruction_arrived)[config::INSTRUCTION_PIPE_STAGES];
    kittens::semaphore (&instruction_finished)[config::INSTRUCTION_PIPE_STAGES];
    kittens::semaphore &instruction_fetch_ready;

    kittens::semaphore &semaphores_ready;

    page_t<config> (&pages)[config::NUM_PAGES];
    kittens::semaphore (&page_finished)[config::NUM_PAGES][config::INSTRUCTION_PIPE_STAGES_BITS];

    int instruction_index;
    int instruction_ring;

    kittens::semaphore &tensor_finished;
    kittens::tensor_allocator<1, 2> &tensor_alloc;

    uint32_t pid_order_shared_addr;

    __device__ inline int (&instruction())[config::INSTRUCTION_WIDTH] {
        return instruction_states[instruction_ring].instruction;
    }
    __device__ inline const int (&instruction() const)[config::INSTRUCTION_WIDTH] {
        return instruction_states[instruction_ring].instruction;
    }
    __device__ inline int (&pid_order())[config::NUM_PAGES] {
        return instruction_states[instruction_ring].pid_order;
    }
    __device__ inline const int (&pid_order() const)[config::NUM_PAGES] {
        return instruction_states[instruction_ring].pid_order;
    }
    __device__ inline void *scratch() const {
        return (void *)&instruction_states[instruction_ring].scratch[0];
    }
    __device__ inline kittens::semaphore (&semaphores())[config::DYNAMIC_SEMAPHORES] {
        return instruction_states[instruction_ring].semaphores;
    }
    __device__ inline const kittens::semaphore (&semaphores() const)[config::DYNAMIC_SEMAPHORES] {
        return instruction_states[instruction_ring].semaphores;
    }

    __device__ inline void await_instruction() {
        kittens::wait(instruction_arrived[instruction_ring], (instruction_index / config::INSTRUCTION_PIPE_STAGES) & 0b1);
        pid_order_shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&(pid_order()[0])));
    }
    __device__ inline void next_instruction() {
        if (kittens::laneid() == 0) // TODO: replace all of these with elect sync
            kittens::arrive(instruction_finished[instruction_ring]);
        instruction_index++;
        instruction_ring = kittens::ring_advance<config::INSTRUCTION_PIPE_STAGES>(instruction_ring);
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

} // namespace megakernel
