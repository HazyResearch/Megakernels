#pragma once

#include "kittens.cuh"

namespace megakittens {

// RMSNorm + QKV matvec + RoPE + KV cache write.
//
// Each SM handles a slice of the 3072 QKV output rows. Every SM redundantly
// normalises the hidden-state vector, then computes its assigned dot products,
// applies RoPE to Q and K, and routes the results to q_post_rope / k_cache / v_cache.
//
// indices[0] = layer_idx
// indices[1] = start_block  (units of BLOCK_SIZE=16 output rows)
// indices[2] = end_block
//
// Scalar globals: pos_id (unsigned int), rms_norm_eps (float)

template <typename Config, typename Globals, int N,
          int SRC0, int SRC1, int SRC2, int SRC3, int SRC4, int SRC5, int SRC6, int DST>
struct RmsQkvRopeAppend {
    // SRC0 = hidden_states     [N]                                         bf16
    // SRC1 = attn_norm_weights [num_layers, N]                             bf16
    // SRC2 = qkv_weights       [num_layers, qkv_dim, N]                   bf16
    // SRC3 = rope_cos          [max_seq_len, head_dim]                     fp32
    // SRC4 = rope_sin          [max_seq_len, head_dim]                     fp32
    // SRC5 = k_cache           [num_layers, max_seq_len, num_kv_heads, head_dim] bf16
    // SRC6 = v_cache           [num_layers, max_seq_len, num_kv_heads, head_dim] bf16
    // DST  = q_post_rope       [q_dim]                                     bf16

    static constexpr int HEAD_DIM = 64;
    static constexpr int BLOCK_SIZE = 16;
    static constexpr int Q_DIM = N;
    static constexpr int ELEMS_PER_LANE = N / kittens::WARP_THREADS;

    __device__ static __forceinline__ float warp_reduce_sum(float val) {
        #pragma unroll
        for (int offset = kittens::WARP_THREADS / 2; offset > 0; offset >>= 1)
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
        return val;
    }

    struct controller {
        __device__ __forceinline__ static int lid_release_order(
            const Globals &g, state_t<Config> &s, int lid) {
            return lid;
        }
        __device__ __forceinline__ static int init_semaphores(
            const Globals &g, state_t<Config> &s) {
            return 0;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            if (kittens::warp::elect_leader()) {
                for (int i = 0; i < Config::NUM_PAGES; i++) {
                    int pid = s.lid_to_pid(i);
                    s.page_wait(pid);
                    s.page_finish(pid);
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
            if (consumer_group::elect_leader())
                all_input_barrier_wait<Config>(g, s.instruction());
            consumer_group::sync(4);

            const auto &inst    = s.instruction();
            const int layer_idx   = inst.indices[0];
            const int start_block = inst.indices[1];
            const int end_block   = inst.indices[2];
            const int warp_id     = kittens::warpid();
            const int lane        = kittens::laneid();

            const kittens::bf16 *hidden   = reinterpret_cast<const kittens::bf16 *>(g.template gls<SRC0>().raw_ptr);
            const kittens::bf16 *norm_w   = reinterpret_cast<const kittens::bf16 *>(g.template gls<SRC1>().raw_ptr);
            const kittens::bf16 *qkv_w    = reinterpret_cast<const kittens::bf16 *>(g.template gls<SRC2>().raw_ptr);
            const float         *rope_cos = reinterpret_cast<const float *>(g.template gls<SRC3>().raw_ptr);
            const float         *rope_sin = reinterpret_cast<const float *>(g.template gls<SRC4>().raw_ptr);
            kittens::bf16       *k_cache  = reinterpret_cast<kittens::bf16 *>(g.template gls<SRC5>().raw_ptr);
            kittens::bf16       *v_cache  = reinterpret_cast<kittens::bf16 *>(g.template gls<SRC6>().raw_ptr);
            kittens::bf16       *q_out    = reinterpret_cast<kittens::bf16 *>(g.template gls<DST>().raw_ptr);

            const int qkv_dim = g.template gls<SRC2>().rows();
            const int k_dim   = (qkv_dim - Q_DIM) / 2;
            const int max_seq = g.template gls<SRC5>().depth();
            const int num_kv  = g.template gls<SRC5>().rows();
            const int pos_id  = g.pos_id;
            const float eps   = g.rms_norm_eps;

            // ── RMSNorm ──────────────────────────────────────────────
            // Each warp redundantly loads + normalises the full hidden vector.
            // Lane j holds elements at indices j, j+32, j+64, ...
            float normed[ELEMS_PER_LANE];
            {
                const kittens::bf16 *layer_norm = norm_w + layer_idx * N;
                float sum_sq = 0.0f;
                #pragma unroll
                for (int i = 0; i < ELEMS_PER_LANE; i++) {
                    const int idx = lane + i * kittens::WARP_THREADS;
                    float val = __bfloat162float(hidden[idx]);
                    normed[i] = val;
                    sum_sq += val * val;
                }
                sum_sq = warp_reduce_sum(sum_sq);
                const float rstd = rsqrtf(sum_sq / static_cast<float>(N) + eps);
                #pragma unroll
                for (int i = 0; i < ELEMS_PER_LANE; i++) {
                    const int idx = lane + i * kittens::WARP_THREADS;
                    normed[i] *= rstd * __bfloat162float(layer_norm[idx]);
                }
            }

            // ── Matvec + RoPE + Store ────────────────────────────────
            const int num_blocks = end_block - start_block;
            const int bpw = (num_blocks + Config::NUM_CONSUMER_WARPS - 1)
                          / Config::NUM_CONSUMER_WARPS;
            const int my_b0 = start_block + warp_id * bpw;
            const int my_b1 = min(my_b0 + bpw, end_block);

            const int64_t layer_w_off =
                static_cast<int64_t>(layer_idx) * qkv_dim * N;
            const float *cos_pos = rope_cos + pos_id * HEAD_DIM;
            const float *sin_pos = rope_sin + pos_id * HEAD_DIM;
            const int64_t kv_layer_off =
                static_cast<int64_t>(layer_idx) * max_seq * num_kv * HEAD_DIM;
            const int kv_token_stride = num_kv * HEAD_DIM;

            for (int blk = my_b0; blk < my_b1; blk++) {
                const int row_base = blk * BLOCK_SIZE;

                for (int p = 0; p < BLOCK_SIZE; p += 2) {
                    const int r0 = row_base + p;
                    const int r1 = r0 + 1;
                    const kittens::bf16 *w0 =
                        qkv_w + layer_w_off + static_cast<int64_t>(r0) * N;
                    const kittens::bf16 *w1 =
                        qkv_w + layer_w_off + static_cast<int64_t>(r1) * N;

                    float d0 = 0.0f, d1 = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < ELEMS_PER_LANE; i++) {
                        const int idx = lane + i * kittens::WARP_THREADS;
                        d0 += __bfloat162float(w0[idx]) * normed[i];
                        d1 += __bfloat162float(w1[idx]) * normed[i];
                    }
                    d0 = warp_reduce_sum(d0);
                    d1 = warp_reduce_sum(d1);

                    if (lane == 0) {
                        const int dim_h = r0 % HEAD_DIM;
                        const float c0 = cos_pos[dim_h];
                        const float s0 = sin_pos[dim_h];
                        const float c1 = cos_pos[dim_h + 1];
                        const float s1 = sin_pos[dim_h + 1];

                        if (r0 < Q_DIM) {
                            // Q — apply RoPE, write to q_post_rope
                            q_out[r0] = __float2bfloat16(d0 * c0 - d1 * s0);
                            q_out[r1] = __float2bfloat16(d1 * c1 + d0 * s1);

                        } else if (r0 < Q_DIM + k_dim) {
                            // K — apply RoPE, write to k_cache
                            const int ki  = r0 - Q_DIM;
                            const int64_t off = kv_layer_off
                                + static_cast<int64_t>(pos_id) * kv_token_stride
                                + (ki / HEAD_DIM) * HEAD_DIM
                                + (ki % HEAD_DIM);
                            k_cache[off]     = __float2bfloat16(d0 * c0 - d1 * s0);
                            k_cache[off + 1] = __float2bfloat16(d1 * c1 + d0 * s1);

                        } else {
                            // V — no RoPE, write to v_cache
                            const int vi  = r0 - Q_DIM - k_dim;
                            const int64_t off = kv_layer_off
                                + static_cast<int64_t>(pos_id) * kv_token_stride
                                + (vi / HEAD_DIM) * HEAD_DIM
                                + (vi % HEAD_DIM);
                            v_cache[off]     = __float2bfloat16(d0);
                            v_cache[off + 1] = __float2bfloat16(d1);
                        }
                    }
                }
            }

            consumer_group::sync(1);
            if (consumer_group::elect_leader()) {
                __threadfence();
                all_barrier_arrive<Config>(g, s.instruction());
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {}
    };
};

} // namespace megakittens
