#pragma once

#include "kittens.cuh"

namespace megakittens {

// Rotary positional embedding with per-token cos/sin gather from a full table.
//
// Inputs:
//   x_gl   : (N, D) bf16  — activations to rotate
//   cos_gl : (max_seq_len, D) bf16 — full cos table (indexed by position)
//   sin_gl : (max_seq_len, D) bf16 — full sin table (indexed by position)
//   pos_gl : (N,) int32            — per-token position index into cos/sin
// Output:
//   y_gl   : (N, D) bf16
//
// For each token n in the instruction's tile:
//   p = pos_gl[n]
//   cos_row = cos_gl[p, :]
//   sin_row = sin_gl[p, :]
//   y[n] = rope(x[n], cos_row, sin_row)
//
// RoPE math (LLaMA broadcast-cos layout where cos[2k] == cos[2k+1]):
//   rotated[2k]   = -x[2k+1]
//   rotated[2k+1] =  x[2k]
//   y = x * cos + rotated * sin
//
// Ported from csrc/reference/qkv_rope_append.cu:consumer — the per-token
// `warp::load(rope_cos_vecs_smem[..], g.rope_cos, {position_id, 0})` pattern,
// stripped of the QKV matmul and KV cache append so it's usable as a
// standalone op. Pair with Gemm (via torch.compile) for the fused QKV path
// until a native fused itype is ported.
//
// Each instruction handles TOKENS_PER_INST = 128 rows × head_dim = 128 cols.
// Pages: 1 for x/y (reuse), 1 scratch for position_ids and the gathered
// cos/sin vectors laid out back-to-back.
template <typename Config, typename Globals, int SRC_X, int SRC_COS, int SRC_SIN, int SRC_POS, int DST_Y>
struct RopeGather {
    static constexpr int HEAD_DIM = 128;
    static constexpr int TOKENS_PER_INST = 128;
    // Pages used: 0 = x/y (reused), 1 = cos scratch, 2 = sin scratch.
    static constexpr int NUM_USED_PAGES = 3;

    using tile_t = kittens::st<kittens::bf16, TOKENS_PER_INST, HEAD_DIM>;

    __device__ static __forceinline__ kittens::semaphore &x_arrived(state_t<Config> &s) { return s.semaphores()[0]; }

    // cos/sin scratch: packed bf16 arrays of shape (TOKENS_PER_INST, HEAD_DIM)
    // = 32 KB each; one full page apiece. We deliberately bypass sv<> here
    // because sv<bf16, 128> has a 192-element (384-byte) padded allocation
    // per row which wouldn't fit 128 rows in a single 32 KB page.
    __device__ static __forceinline__ kittens::bf16 *cos_scratch(state_t<Config> &s) {
        return reinterpret_cast<kittens::bf16 *>(&s.pages[s.lid_to_pid(1)]);
    }
    __device__ static __forceinline__ kittens::bf16 *sin_scratch(state_t<Config> &s) {
        return reinterpret_cast<kittens::bf16 *>(&s.pages[s.lid_to_pid(2)]);
    }
    // Position ids in the per-instruction scratch buffer (not a page).
    __device__ static __forceinline__ int *pos_scratch(state_t<Config> &s) {
        return reinterpret_cast<int *>(s.scratch());
    }

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            // Release unused pages (3..6) first, then cos/sin scratch (1, 2), then x/y (0).
            constexpr int order[] = {3, 4, 5, 6, 1, 2, 0};
            return order[query];
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            if (kittens::laneid() == 0) kittens::init_semaphore(x_arrived(s), 1);
            return 1;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const auto &x_gl   = g.template gls<SRC_X>();
            const int tile_row = instruction.indices[0];
            const int tile_col = instruction.indices[1];

            if (kittens::warp::elect_leader()) {
                all_input_barrier_wait<Config>(g, instruction);
                s.page_wait(s.lid_to_pid(0));   // x/y
                s.page_wait(s.lid_to_pid(1));   // cos scratch
                s.page_wait(s.lid_to_pid(2));   // sin scratch
                tile_t &x_st = s.pages[s.lid_to_pid(0)].template as<tile_t>();
                kittens::tma::expect_bytes(x_arrived(s), sizeof(tile_t));
                kittens::tma::load_async(x_st, x_gl, {tile_row, tile_col}, x_arrived(s));
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
            const auto &cos_gl = g.template gls<SRC_COS>();
            const auto &sin_gl = g.template gls<SRC_SIN>();
            const auto &pos_gl = g.template gls<SRC_POS>();
            auto &y_gl = g.template gls<DST_Y>();
            const int tile_row = instruction.indices[0];
            const int tile_col = instruction.indices[1];

            kittens::wait(x_arrived(s), 0);

            tile_t &x_st = s.pages[s.lid_to_pid(0)].template as<tile_t>();
            kittens::bf16 *cos_rows = cos_scratch(s);  // flat (TOKENS_PER_INST * HEAD_DIM) bf16
            kittens::bf16 *sin_rows = sin_scratch(s);

#define ROPE_GATHER_STAGE 1  // 1 = pos only, 2 = pos+gather, 3 = full


            constexpr int ROWS_PER_WARP = TOKENS_PER_INST / Config::NUM_CONSUMER_WARPS;
            static_assert(ROWS_PER_WARP > 0, "Not enough rows for the consumer warps");
            static_assert(HEAD_DIM % 32 == 0, "head_dim must be divisible by warp size");
            constexpr int ELTS_PER_LANE = HEAD_DIM / 32;

#if ROPE_GATHER_STAGE >= 1
            // Diagnostic: single-thread read of pos_ids.
            if (consumer_group::elect_leader()) {
                const int *pos_ptr = reinterpret_cast<const int *>(pos_gl.raw_ptr);
                int *pos_dst = pos_scratch(s);
                for (int i = 0; i < TOKENS_PER_INST; i++) {
                    pos_dst[i] = pos_ptr[i];
                }
            }
            consumer_group::sync(1);
#endif

#if ROPE_GATHER_STAGE <= 1
            // Early exit — store x unchanged.
            if (consumer_group::elect_leader()) {
                all_reuse_barrier_wait<Config>(g, instruction);
                kittens::tma::store_async(y_gl, x_st, {tile_row, tile_col});
                kittens::tma::store_async_wait();
                s.page_finish(s.lid_to_pid(0));
                s.page_finish(s.lid_to_pid(1));
                s.page_finish(s.lid_to_pid(2));
                all_barrier_arrive<Config>(g, instruction);
            }
            return;
#endif

            // Phase 1: each warp gathers its tokens' cos/sin vectors from global memory
            // into the shared scratch pages. Use direct pointer arithmetic — each lane
            // moves 8 bf16 (16 bytes) per iter; HEAD_DIM=128 ⇒ 16 bytes/lane covers the row.
            const int warp_base = kittens::warpid() * ROWS_PER_WARP;
            const kittens::bf16 *cos_base = reinterpret_cast<const kittens::bf16 *>(cos_gl.raw_ptr);
            const kittens::bf16 *sin_base = reinterpret_cast<const kittens::bf16 *>(sin_gl.raw_ptr);
            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                const int token = warp_base + r;
                const int pos = pos_scratch(s)[token];
                const kittens::bf16 *cos_src = cos_base + pos * HEAD_DIM;
                const kittens::bf16 *sin_src = sin_base + pos * HEAD_DIM;
                kittens::bf16 *cos_dst = cos_rows + token * HEAD_DIM;
                kittens::bf16 *sin_dst = sin_rows + token * HEAD_DIM;
                const int per_lane = HEAD_DIM / 32;  // = 4
                const int lane = kittens::laneid();
                #pragma unroll
                for (int j = 0; j < per_lane; j++) {
                    cos_dst[lane * per_lane + j] = cos_src[lane * per_lane + j];
                    sin_dst[lane * per_lane + j] = sin_src[lane * per_lane + j];
                }
            }
            consumer_group::sync(2);

            // Phase 2: apply RoPE per-token, per-lane. Use st::operator[]({row,col})
            // for x (so the swizzled index math matches how TMA wrote it). cos/sin
            // scratch is a plain row-major bf16 array we wrote ourselves.
            const int lane = kittens::laneid();
            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                const int row = warp_base + r;
                float xv[ELTS_PER_LANE], cv[ELTS_PER_LANE], sv_[ELTS_PER_LANE];
                #pragma unroll
                for (int j = 0; j < ELTS_PER_LANE; j++) {
                    const int col = lane * ELTS_PER_LANE + j;
                    xv[j]  = __bfloat162float(x_st[{row, col}]);
                    cv[j]  = __bfloat162float(cos_rows[row * HEAD_DIM + col]);
                    sv_[j] = __bfloat162float(sin_rows[row * HEAD_DIM + col]);
                }
                float rot[ELTS_PER_LANE];
                #pragma unroll
                for (int j = 0; j < ELTS_PER_LANE; j++) {
                    float swapped = __shfl_xor_sync(0xffffffff, xv[j], 1);
                    rot[j] = (lane % 2 == 0) ? -swapped : swapped;
                }
                #pragma unroll
                for (int j = 0; j < ELTS_PER_LANE; j++) {
                    const int col = lane * ELTS_PER_LANE + j;
                    const float y = xv[j] * cv[j] + rot[j] * sv_[j];
                    x_st[{row, col}] = __float2bfloat16(y);
                }
            }
            consumer_group::sync(3);

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
