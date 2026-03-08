#pragma once

#include "kittens.cuh"

#include "../util.cuh"

namespace megakernel {
namespace controller {

template <typename config, typename globals>
__device__ inline bool load_instructions(int *instruction,
                                         int instruction_index,
                                         const globals &g) {
    static_assert(config::INSTRUCTION_WIDTH <= 32);
    auto laneid = ::kittens::laneid();
    int *src_ptr;
    if constexpr (kittens::ducks::gl::all<decltype(g.instructions)>) {
        if constexpr (config::ENABLE_GLOBAL_WORK_QUEUE) {
            src_ptr = &g.instructions[kittens::coord<>{instruction_index, 0}];
        }
        else {
            src_ptr = &g.instructions[kittens::coord<>{(int)(get_worker_id()), instruction_index, 0}];
        }
        static_assert(std::is_same<decltype(src_ptr), int *>::value, "src_ptr is not an int*");
        if (laneid < config::INSTRUCTION_WIDTH)
            instruction[laneid] = src_ptr[laneid];
    } else if constexpr (kittens::ducks::pgl::all<decltype(g.instructions)>) {
        if constexpr (config::ENABLE_GLOBAL_WORK_QUEUE) {
            src_ptr = &g.instructions[g.dev_idx][kittens::coord<>{instruction_index, 0}];
        }
        else {
            src_ptr = &g.instructions[g.dev_idx][kittens::coord<>{(int)(get_worker_id()), instruction_index, 0}];
        }
        static_assert(std::is_same<decltype(src_ptr), int *>::value, "src_ptr is not an int*");
        if (laneid < config::INSTRUCTION_WIDTH)
            instruction[laneid] = src_ptr[laneid];
    }
    kittens::warp::sync();
    return instruction[0] != -1;
}

} // namespace controller
} // namespace megakernel
