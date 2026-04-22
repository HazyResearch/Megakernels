#pragma once

#include "kittens.cuh"

namespace megakittens {
namespace llama70b {

template <typename Config, int N>
__device__ static inline auto rms_norm(
        kittens::rv_fl<N / Config::NUM_CONSUMER_WARPS> activations_vec,
        const kittens::sv_bf<N / Config::NUM_CONSUMER_WARPS> &rms_scale_smem,
        float rms_norm_eps, float *scratch_memory) {
    constexpr int ELEMS_PER_WARP = N / Config::NUM_CONSUMER_WARPS;
    using rv_t = kittens::rv_fl<ELEMS_PER_WARP>;
    rv_t sq_activations_vec, rms_scale_vec;
    kittens::warp::copy(sq_activations_vec, activations_vec);
    kittens::warp::mul(sq_activations_vec, sq_activations_vec, sq_activations_vec);
    float partial_sum = kittens::warp::sum(sq_activations_vec);

    if (kittens::warp::elect_leader()) {
        scratch_memory[kittens::warpid()] = partial_sum;
    }
    kittens::group<Config::NUM_CONSUMER_WARPS>::sync(1);

    float full_sum = 0.f;
    #pragma unroll
    for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++) {
        full_sum += scratch_memory[i];
    }

    float variance = full_sum / static_cast<float>(N);
    float rms_scale = rsqrtf(variance + rms_norm_eps);

    kittens::warp::load(rms_scale_vec, rms_scale_smem);
    kittens::warp::mul(rms_scale_vec, rms_scale_vec, rms_scale);
    kittens::warp::mul(activations_vec, activations_vec, rms_scale_vec);

    return activations_vec;
}

} // namespace llama70b
} // namespace megakittens
