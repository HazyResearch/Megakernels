#pragma once

#include "kittens.cuh"

namespace megakittens {

// Rotary positional embedding on bf16 activations.
//
// Inputs:
//   x_gl   : (N, D) bf16 — activations to rotate
//   cos_gl : (N, D) bf16 — per-row cos, already gathered at position_ids
//   sin_gl : (N, D) bf16 — per-row sin, already gathered at position_ids
// Output:
//   y_gl   : (N, D) bf16
//
// RoPE math (matches LLaMA's broadcast-cos layout where cos[n, 2k] == cos[n, 2k+1]):
//   rotated[n, 2k]   = -x[n, 2k+1]
//   rotated[n, 2k+1] =  x[n, 2k]
//   y = x * cos + rotated * sin
//
// Ported from csrc/reference/qkv_rope_append.cu `apply_rope_inplace`, stripped
// of the QKV matmul + KV cache append. Callers that want the full fused op
// can compose Gemm + Rope through torch.compile.
//
// Each instruction handles TOKENS_PER_INST = 128 rows × head_dim = 128 cols.
// 128 tokens × 128 dims × 2 bytes = 32 KB = one page, so we use 4 pages total
// (x, cos, sin, y; y reuses x's page after the rotation is consumed).
template <typename Config, typename Globals, int SRC_X, int SRC_COS, int SRC_SIN, int DST_Y>
struct Rope {
    static constexpr int HEAD_DIM = 128;
    static constexpr int TOKENS_PER_INST = 128;
    static constexpr int NUM_USED_PAGES = 3;

    using tile_t = kittens::st<kittens::bf16, TOKENS_PER_INST, HEAD_DIM>;

    __device__ static __forceinline__ kittens::semaphore &inputs_arrived(state_t<Config> &s) {
        return s.semaphores()[0];
    }

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            // Release the cos/sin pages first (lids 1, 2), then x/out page (lid 0 reused).
            constexpr int order[] = {1, 2, 0, 3, 4, 5, 6};
            return order[query];
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            if (kittens::laneid() == 0) kittens::init_semaphore(inputs_arrived(s), 1);
            return 1;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const auto &x_gl   = g.template gls<SRC_X>();
            const auto &cos_gl = g.template gls<SRC_COS>();
            const auto &sin_gl = g.template gls<SRC_SIN>();
            const int tile_row = instruction.indices[0];
            const int tile_col = instruction.indices[1];

            if (kittens::warp::elect_leader()) {
                all_input_barrier_wait<Config>(g, instruction);

                const int x_pid   = s.lid_to_pid(0);
                const int cos_pid = s.lid_to_pid(1);
                const int sin_pid = s.lid_to_pid(2);
                s.page_wait(x_pid);
                s.page_wait(cos_pid);
                s.page_wait(sin_pid);
                tile_t &x_st   = s.pages[x_pid].template as<tile_t>();
                tile_t &cos_st = s.pages[cos_pid].template as<tile_t>();
                tile_t &sin_st = s.pages[sin_pid].template as<tile_t>();

                kittens::tma::expect_bytes(inputs_arrived(s), 3 * sizeof(tile_t));
                kittens::tma::load_async(x_st,   x_gl,   {tile_row, tile_col}, inputs_arrived(s));
                kittens::tma::load_async(cos_st, cos_gl, {tile_row, tile_col}, inputs_arrived(s));
                kittens::tma::load_async(sin_st, sin_gl, {tile_row, tile_col}, inputs_arrived(s));
            } else if (kittens::warp::elect_leader_from_active()) {
                #pragma unroll
                for (int i = NUM_USED_PAGES; i < Config::NUM_PAGES; i++) {
                    s.page_wait(s.lid_to_pid(i));
                    s.page_finish(s.lid_to_pid(i));
                }
            }
        }
    };

    struct launcher {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            s.tensor_wait();
            if (kittens::warp::elect_leader()) s.tensor_finish();
        }
    };

    struct consumer {
        using consumer_group = kittens::group<Config::NUM_CONSUMER_WARPS>;

        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            auto &y_gl = g.template gls<DST_Y>();
            const int tile_row = instruction.indices[0];
            const int tile_col = instruction.indices[1];

            kittens::wait(inputs_arrived(s), 0);

            tile_t &x_st   = s.pages[s.lid_to_pid(0)].template as<tile_t>();
            tile_t &cos_st = s.pages[s.lid_to_pid(1)].template as<tile_t>();
            tile_t &sin_st = s.pages[s.lid_to_pid(2)].template as<tile_t>();

            // Each of NUM_CONSUMER_WARPS warps handles TOKENS_PER_INST / NUM_CONSUMER_WARPS rows.
            constexpr int ROWS_PER_WARP = TOKENS_PER_INST / Config::NUM_CONSUMER_WARPS;
            static_assert(ROWS_PER_WARP > 0, "Not enough rows for the consumer warps");
            static_assert(HEAD_DIM % 32 == 0, "head_dim must be divisible by warp size (32)");
            constexpr int ELTS_PER_LANE = HEAD_DIM / 32;

            using rv_t = kittens::rv_bf<HEAD_DIM>;
            rv_t x_vec, cos_vec, sin_vec, rotated;

            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                const int row = kittens::warpid() * ROWS_PER_WARP + r;

                kittens::warp::load(x_vec,   x_st,   {row, 0});
                kittens::warp::load(cos_vec, cos_st, {row, 0});
                kittens::warp::load(sin_vec, sin_st, {row, 0});
                kittens::warp::sync();

                #pragma unroll
                for (int i = 0; i < ELTS_PER_LANE; i++) {
                    // Convert to float for the shuffle so the adjacent-lane swap works on a
                    // full 32-bit value, then cast back to bf16.
                    float xi = __bfloat162float(x_vec[{i, 0}]);
                    float swapped = __shfl_xor_sync(0xffffffff, xi, 1);
                    if (kittens::laneid() % 2 == 0) swapped = -swapped;
                    rotated[{i, 0}] = __float2bfloat16(swapped);
                }
                kittens::warp::sync();

                kittens::warp::mul(x_vec,   x_vec,   cos_vec);
                kittens::warp::mul(rotated, rotated, sin_vec);
                kittens::warp::add(x_vec,   x_vec,   rotated);

                kittens::warp::store(x_st, x_vec, {row, 0});
            }

            consumer_group::sync(1);

            if (consumer_group::elect_leader()) {
                all_reuse_barrier_wait<Config>(g, instruction);
                kittens::tma::store_async(y_gl, x_st, {tile_row, tile_col});
                kittens::tma::store_async_wait();
                s.page_finish(s.lid_to_pid(0));
                s.page_finish(s.lid_to_pid(1));
                s.page_finish(s.lid_to_pid(2));
                all_barrier_arrive<Config>(g, instruction);
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) { }
    };
};

} // namespace megakittens
