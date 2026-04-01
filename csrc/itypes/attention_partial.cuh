#pragma once

#include "kittens.cuh"

namespace megakittens {

// Naive single-partition decode attention.
// Computes: for each q_head in [q_head_start, q_head_start + GQA_RATIO):
//   attn_out[q_head] = softmax(q[q_head] @ k_cache[layer, :seq_len, kv_head].T * scale) @ v_cache[layer, :seq_len, kv_head]
//
// SRC0 = q_post_rope   [NUM_Q_HEADS * HEAD_DIM] bf16  (flat vector)
// SRC1 = k_cache        [NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM] bf16
// SRC2 = v_cache        [NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM] bf16
// DST  = attn_out       [NUM_Q_HEADS * HEAD_DIM] bf16  (flat vector)
//
// Scalars on globals: pos_id (unsigned int), attn_scale (float)

template <typename Config, typename Globals, int SRC0, int SRC1, int SRC2, int DST>
struct AttentionPartial {
    static constexpr int GQA_RATIO = 4;
    static constexpr int HEAD_DIM = 64;

    struct parsed_instruction {
        int layer_idx;
        int kv_head_idx;
        __device__ inline parsed_instruction(state_t<Config> &s) {
            const auto &inst = s.instruction();
            layer_idx   = inst.indices[0];
            kv_head_idx = inst.indices[1];
        }
    };

    __device__ static __forceinline__ float warp_reduce_sum(float val) {
        #pragma unroll
        for (int offset = kittens::WARP_THREADS / 2; offset > 0; offset >>= 1)
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
        return val;
    }

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int lid) {
            return lid;
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
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

            parsed_instruction inst{s};
            const int seq_len = g.pos_id + 1;
            const float scale = g.attn_scale;
            const int warp_id = kittens::warpid();
            const int lane    = kittens::laneid();

            const kittens::bf16 *q_ptr   = reinterpret_cast<const kittens::bf16 *>(g.template gls<SRC0>().raw_ptr);
            const kittens::bf16 *k_cache = reinterpret_cast<const kittens::bf16 *>(g.template gls<SRC1>().raw_ptr);
            const kittens::bf16 *v_cache = reinterpret_cast<const kittens::bf16 *>(g.template gls<SRC2>().raw_ptr);
            kittens::bf16       *out     = reinterpret_cast<kittens::bf16 *>(g.template gls<DST>().raw_ptr);

            // k/v cache dims: [NUM_LAYERS, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM]
            const int max_seq_len  = g.template gls<SRC1>().depth();
            const int num_kv_heads = g.template gls<SRC1>().rows();
            const int64_t kv_stride_layer = static_cast<int64_t>(max_seq_len) * num_kv_heads * HEAD_DIM;
            const int     kv_stride_token = num_kv_heads * HEAD_DIM;

            if (warp_id < GQA_RATIO) {
                const int q_head = inst.kv_head_idx * GQA_RATIO + warp_id;
                const kittens::bf16 *q = q_ptr + q_head * HEAD_DIM;

                // Base pointer into this layer+kv_head slice of the cache
                const kittens::bf16 *k_base = k_cache + inst.layer_idx * kv_stride_layer + inst.kv_head_idx * HEAD_DIM;
                const kittens::bf16 *v_base = v_cache + inst.layer_idx * kv_stride_layer + inst.kv_head_idx * HEAD_DIM;

                // Load q into registers (2 elements per lane for HEAD_DIM=64, 32 lanes)
                float q_reg[2];
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    int idx = lane + i * kittens::WARP_THREADS;
                    q_reg[i] = (idx < HEAD_DIM) ? __bfloat162float(q[idx]) : 0.0f;
                }

                // Online softmax state
                float max_score = kittens::base_types::constants<float>::neg_infty();
                float sum_exp = 0.0f;
                float o_reg[2] = {0.0f, 0.0f};

                for (int t = 0; t < seq_len; t++) {
                    const kittens::bf16 *k_vec = k_base + static_cast<int64_t>(t) * kv_stride_token;

                    // dot(q, k) * scale
                    float dot = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 2; i++) {
                        int idx = lane + i * kittens::WARP_THREADS;
                        if (idx < HEAD_DIM)
                            dot += q_reg[i] * __bfloat162float(k_vec[idx]);
                    }
                    float score = warp_reduce_sum(dot) * scale;

                    // Online softmax update
                    float old_max = max_score;
                    max_score = fmaxf(max_score, score);
                    float rescale = expf(old_max - max_score);
                    sum_exp = sum_exp * rescale + expf(score - max_score);

                    #pragma unroll
                    for (int i = 0; i < 2; i++)
                        o_reg[i] *= rescale;

                    // Accumulate v * exp(score - max)
                    const kittens::bf16 *v_vec = v_base + static_cast<int64_t>(t) * kv_stride_token;
                    float w = expf(score - max_score);
                    #pragma unroll
                    for (int i = 0; i < 2; i++) {
                        int idx = lane + i * kittens::WARP_THREADS;
                        if (idx < HEAD_DIM)
                            o_reg[i] += w * __bfloat162float(v_vec[idx]);
                    }
                }

                // Normalize and write output
                float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    int idx = lane + i * kittens::WARP_THREADS;
                    if (idx < HEAD_DIM)
                        out[q_head * HEAD_DIM + idx] = __float2bfloat16(o_reg[i] * inv_sum);
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
