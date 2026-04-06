#pragma once

#include "kittens.cuh"

namespace megakittens {

enum WorkerType {
    page_manager = 0,
    semaphore_manager = 1,
    loader = 2,
    launcher = 3,
    consumer = 4,
    storer = 5
};

template <typename IType>
concept MegaKittensIType = requires {
    typename IType::controller;
    typename IType::loader;
    typename IType::launcher;
    typename IType::consumer;
    typename IType::storer;
};

struct instruction_t {
    static constexpr int MAX_SRC_TENSORS = 16;
    static constexpr int MAX_DST_TENSORS = 8;
    static constexpr int MAX_INDICES = 16;
    static constexpr int MAX_SRC_BARRIERS = 16;
    static constexpr int MAX_DST_BARRIERS = 8;

    int icode;                                  //  4B
    uint8_t src_tensors[MAX_SRC_TENSORS];       // 16B
    uint8_t dst_tensors[MAX_DST_TENSORS];       //  8B
    int indices[MAX_INDICES];                   // 64B
    uint32_t src_barriers[MAX_SRC_BARRIERS];    // 64B
    int src_barrier_targets[MAX_SRC_BARRIERS];  // 64B
    uint8_t num_input_barriers;                 //  1B
    uint8_t num_reuse_barriers;                 //  1B
    uint8_t num_dst_barriers;                   //  1B
    uint8_t _;                                  //  1B (padding)
    uint32_t dst_barriers[MAX_DST_BARRIERS];    // 32B
};
static_assert(sizeof(instruction_t) == 256);

struct default_config {
    static constexpr int INSTRUCTION_PIPE_STAGES = 2;
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int MIN_BLOCKS_PER_SM = 1;
    static_assert(INSTRUCTION_PIPE_STAGES == 2 && CLUSTER_SIZE == 2 && MIN_BLOCKS_PER_SM == 1); // should not change

    static constexpr int NUM_CONSUMER_WARPS = 8;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
    static constexpr int CONSUMER_REGISTERS = 224;
    static constexpr int NON_CONSUMER_REGISTERS = 56;

    static constexpr int DYNAMIC_SEMAPHORES = 32;
    static_assert(DYNAMIC_SEMAPHORES <= 32); // for warp parallel processing

    static constexpr int PAGE_SIZE = 32768; // this is the only knob and everything else is derived
    static constexpr int STATIC_SHARED_MEMORY_BASE = 512 + INSTRUCTION_PIPE_STAGES*(sizeof(instruction_t) + 128 + DYNAMIC_SEMAPHORES*8);
    static constexpr int DYNAMIC_SHARED_MEMORY_ALIGN = 1024; // alignment overhead
    static constexpr int NUM_PAGES = (kittens::MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY_BASE - DYNAMIC_SHARED_MEMORY_ALIGN) / PAGE_SIZE;
    static_assert(NUM_PAGES >= 1 && NUM_PAGES <= 32); // for warp parallel processing and instruction_state_t padding
    static constexpr int DYNAMIC_SHARED_MEMORY = NUM_PAGES * PAGE_SIZE + DYNAMIC_SHARED_MEMORY_ALIGN;
    static constexpr int SCRATCH_BYTES = (kittens::MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY_BASE - DYNAMIC_SHARED_MEMORY) / INSTRUCTION_PIPE_STAGES;
    static constexpr int STATIC_SHARED_MEMORY = STATIC_SHARED_MEMORY_BASE + INSTRUCTION_PIPE_STAGES*SCRATCH_BYTES;

    static constexpr int SPIN_LOOP_SLEEP_NS = 20;
    static constexpr int TIMING_WIDTH = 16; // # of int32s for timing
};

template <typename Config>
struct __align__(128) instruction_state_t {
    instruction_t instruction;
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
    template <typename T>
    __device__ __forceinline__ T &as(int byte_offset = 0) {
        static_assert(sizeof(T) <= Config::PAGE_SIZE, "T exceeds page size"); // only guarantees safety for byte_offset=0
        return *reinterpret_cast<T*>(reinterpret_cast<uint64_t>(&data[0]) + byte_offset);
    }
    template <typename T>
    __device__ __forceinline__ const T &as(int byte_offset = 0) const {
        static_assert(sizeof(T) <= Config::PAGE_SIZE, "T exceeds page size"); // only guarantees safety for byte_offset=0
        return *reinterpret_cast<const T*>(reinterpret_cast<uint64_t>(&data[0]) + byte_offset);
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
    kittens::tensor_allocator<1, Config::CLUSTER_SIZE> &tensor_alloc;

    // timings_ptr is nullptr if profiling is disabled
    int *timings_ptr;
    int timings_stride;
    uint64_t start_clock;

    // called once per instruction to record instruction type into slot 0 (special case)
    __device__ __forceinline__ void record(int event_id, int value) {
        if (timings_ptr != nullptr) {
            int offset = iter * Config::TIMING_WIDTH + event_id;
            if (offset < timings_stride)
                timings_ptr[blockIdx.x * timings_stride + offset] = value;
        }
    }
    // write timestamp into given event slot index
    __device__ __forceinline__ void record(int event_id) {
        record(event_id, (int)(clock64() - start_clock));
    }

    __device__ __forceinline__ const instruction_t &instruction() const {
        return instruction_states[stage].instruction;
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
    __device__ __forceinline__ int lid_to_pid(int lid) {
        return pid_order()[lid];
    }
    __device__ __forceinline__ void page_wait(int pid) {
        kittens::wait(page_finished[pid], iter&0b1);
    }
    __device__ __forceinline__ void page_finish(int pid) {
        kittens::arrive(page_finished[pid]);
    }
    __device__ __forceinline__ void tensor_wait() {
        kittens::wait(tensor_finished, iter&0b1);
    }
    __device__ __forceinline__ void tensor_finish() {
        kittens::arrive(tensor_finished);
    }
};

} // namespace megakittens
