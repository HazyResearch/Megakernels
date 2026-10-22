#pragma once

#include "kittens.cuh"

namespace megakernel {

struct default_config {
    // Instruction pipeline
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2;

    // num bits required to represent num pipeline stages
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 1;

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
#if defined(KITTENS_5090)
    // GB202 has the Blackwell instruction set but the consumer-GPU shared
    // memory capacity. ThunderKittens' legacy branch otherwise reports the
    // datacenter Blackwell/Hopper capacity when both feature macros are set.
    static constexpr int MAX_SHARED_MEMORY = 100000;
#else
    static constexpr int MAX_SHARED_MEMORY = ::kittens::MAX_SHARED_MEMORY;
#endif

    // Shared memory declared statically
    static constexpr int SCRATCH_BYTES = 4096;
    static constexpr int STATIC_SHARED_MEMORY =
        512 + INSTRUCTION_PIPELINE_STAGES *
                  (SCRATCH_BYTES + (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 +
                   DYNAMIC_SEMAPHORES * 8);
#if defined(KITTENS_5090)
    static constexpr int AVAILABLE_DYNAMIC_SHARED_MEMORY =
        MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;
#else
    static constexpr int AVAILABLE_DYNAMIC_SHARED_MEMORY =
        ::kittens::MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;
#endif

    // Shared memory declared dynamically
    static constexpr int PAGE_SIZE = 16384;
    static constexpr int PAGE_ALIGNMENT = 1024;
    static constexpr int NUM_PAGES =
        (AVAILABLE_DYNAMIC_SHARED_MEMORY - (PAGE_ALIGNMENT - 1)) / PAGE_SIZE;
    static constexpr int DYNAMIC_SHARED_MEMORY =
        NUM_PAGES * PAGE_SIZE + (PAGE_ALIGNMENT - 1);
    static_assert(DYNAMIC_SHARED_MEMORY <= AVAILABLE_DYNAMIC_SHARED_MEMORY,
                  "Dynamic pages and alignment padding exceed shared memory");
    static_assert(NUM_PAGES >= 5,
                  "Megakernel VM requires at least five shared-memory pages");
    static_assert(NUM_PAGES <= 32,
                  "Page allocator uses one warp-wide bit mask");

    static constexpr bool TIMING_RECORD_ENABLED = false;

    static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS = 20;

    static constexpr int CONSUMER_REGISTERS = 104;
    static constexpr int NON_CONSUMER_REGISTERS = 64;
};
template <typename config>
using instruction_layout = kittens::gl<int, 1, -1, -1, config::INSTRUCTION_WIDTH>;
template <typename config>
using timing_layout = kittens::gl<int, 1, -1, -1, config::TIMING_WIDTH>;

template <typename config> void print_config() {
    std::cout << "---------------- CONFIG INFO ----------------" << std::endl;
    std::cout << "INSTRUCTION_PIPELINE_STAGES: "
              << config::INSTRUCTION_PIPELINE_STAGES << std::endl;
    std::cout << "INSTRUCTION_WIDTH: " << config::INSTRUCTION_WIDTH
              << std::endl;
    std::cout << "TIMING_WIDTH: " << config::TIMING_WIDTH << std::endl;
    std::cout << "NUM_CONSUMER_WARPS: " << config::NUM_CONSUMER_WARPS
              << std::endl;
    std::cout << "NUM_WARPS: " << config::NUM_WARPS << std::endl;
    std::cout << "NUM_THREADS: " << config::NUM_THREADS << std::endl;
    std::cout << "NUM_BLOCKS: " << config::NUM_BLOCKS << std::endl;
    std::cout << "CLUSTER_BLOCKS: " << config::CLUSTER_BLOCKS << std::endl;
    std::cout << "MAX_SHARED_MEMORY: " << config::MAX_SHARED_MEMORY
              << std::endl;
    std::cout << "STATIC_SHARED_MEMORY: " << config::STATIC_SHARED_MEMORY
              << std::endl;
    std::cout << "PAGE_SIZE: " << config::PAGE_SIZE << std::endl;
    std::cout << "NUM_PAGES: " << config::NUM_PAGES << std::endl;
    std::cout << "SCRATCH_BYTES: " << config::SCRATCH_BYTES << std::endl;
    std::cout << "DYNAMIC_SEMAPHORES: " << config::DYNAMIC_SEMAPHORES
              << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
}

} // namespace megakernel