#pragma once

#include "kittens.cuh"

namespace megakittens {
namespace llama1b {

template <typename Config, int N>
__device__ static inline auto
rms_norm(const kittens::sv_bf<N / Config::NUM_CONSUMER_WARPS> &rms_scale_smem,
         const kittens::sv_bf<N / Config::NUM_CONSUMER_WARPS> &activations_smem,
         float rms_norm_eps, float *scratch_memory) {

    constexpr int ELEMS_PER_WARP = N / Config::NUM_CONSUMER_WARPS;
    using rv_t = kittens::rv_fl<ELEMS_PER_WARP>;
    rv_t activations_vec, sq_activations_vec, rms_scale_vec;

    kittens::warp::load(activations_vec, activations_smem);
    kittens::warp::copy(sq_activations_vec, activations_vec);
    kittens::warp::mul(sq_activations_vec, sq_activations_vec, sq_activations_vec);
    float partial_sum = kittens::warp::sum(sq_activations_vec);

    if (kittens::warp::elect_leader()) {
        scratch_memory[kittens::warpid()] = partial_sum;
    }
    kittens::group<Config::NUM_CONSUMER_WARPS>::sync(0);

    float full_sum = 0.f;
    #pragma unroll
    for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++) {
        full_sum += scratch_memory[i];
    }

    float variance = full_sum / static_cast<float>(N);
    float rms_scale = rsqrtf(variance + rms_norm_eps);

    kittens::warp::mul(activations_vec, activations_vec, rms_scale);
    kittens::warp::load(rms_scale_vec, rms_scale_smem);
    kittens::warp::mul(activations_vec, activations_vec, rms_scale_vec);

    return activations_vec;
}

#ifdef KITTENS_BLACKWELL
template <kittens::ducks::st::all st_t>
__device__ static inline void matvec(kittens::sv_fl<st_t::rows> &out_smem,
                                     st_t &weights_smem,
                                     kittens::rv_fl<st_t::cols> &activations) {
    using rt_t  = kittens::rt_bf<st_t::rows, st_t::cols>;
    using rrv_t = typename rt_t::row_vec;

    rrv_t row_activations;
    kittens::warp::copy(row_activations, activations);

    rt_t broadcast_activations, weights;
    kittens::warp::broadcast_col(broadcast_activations, row_activations);
    kittens::warp::load(weights, weights_smem);

    kittens::rt_fl<16, 16> out_activations;
    kittens::warp::zero(out_activations);
    kittens::warp::mma_ABt(out_activations, weights, broadcast_activations, out_activations);

    if (kittens::laneid() % 4 == 0) {
        int row0 = kittens::laneid() / 4;
        int row1 = row0 + 8;
        out_smem[row0] = out_activations.tiles[0][0].data[0].x;
        out_smem[row1] = out_activations.tiles[0][0].data[1].x;
    }
    kittens::warp::sync();
}
#else
template <kittens::ducks::st::all st_t>
__device__ static inline void matvec(kittens::sv_fl<st_t::rows> &out_smem,
                                     st_t &weights_smem,
                                     kittens::rv_fl<st_t::cols> &activations) {
    using rt_t  = kittens::rt_fl<st_t::rows, st_t::cols>;
    using rrv_t = typename rt_t::row_vec;
    using rcv_t = typename rt_t::col_vec;
    using rv_t  = kittens::rv_fl<st_t::rows>;

    rrv_t row_activations;
    kittens::warp::copy(row_activations, activations);

    rt_t broadcast_activations, weights;
    kittens::warp::broadcast_col(broadcast_activations, row_activations);
    kittens::warp::load(weights, weights_smem);
    kittens::warp::mul(broadcast_activations, broadcast_activations, weights);

    rcv_t sum_col_vec;
    kittens::warp::row_sum(sum_col_vec, broadcast_activations);

    rv_t sum_vec;
    kittens::warp::copy(sum_vec, sum_col_vec);

    if (kittens::laneid() < st_t::rows) {
        out_smem[kittens::laneid()] = sum_vec[0][0];
    }
    kittens::warp::sync();
}
#endif

template <typename Config, int SCRATCH_BYTES_PER_WARP>
__device__ static inline void matvec_reduce(uint8_t *scratch, kittens::rv_fl<16> &sum_vec) {
    using sv_t = kittens::sv_fl<16>;
    kittens::rv_fl<16> part_vec;
    kittens::warp::zero(sum_vec);

    #pragma unroll
    for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++) {
        sv_t &part = *reinterpret_cast<sv_t *>(scratch + i * SCRATCH_BYTES_PER_WARP);
        kittens::warp::load(part_vec, part);
        kittens::warp::add(sum_vec, sum_vec, part_vec);
    }
}

} // namespace llama1b
} // namespace megakittens
