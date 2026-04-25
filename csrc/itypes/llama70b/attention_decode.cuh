#pragma once

#include "kittens.cuh"
#include "schema.cuh"

namespace megakittens {
namespace llama70b {

template <typename Config, typename Globals,
          int BATCH_SIZE, int NUM_PAGES, int PAGES_PER_SEQ,
          int HEAD_DIM, int PAGE_SIZE, int KV_BLOCK_SIZE,
          int NUM_Q_HEADS, int NUM_KV_HEADS,
          int SRC_Q, int SRC_K_CACHE, int SRC_V_CACHE, int SCALAR_POS_ID, int SCALAR_ATTN_SCALE,
          int DST_OUT>
struct AttentionDecode {
    static_assert(HEAD_DIM == 128, "AttentionDecode70b expects head_dim=128.");
    static_assert(NUM_Q_HEADS == 64, "AttentionDecode70b expects 64 Q heads.");
    static_assert(NUM_KV_HEADS == 8, "AttentionDecode70b expects 8 KV heads.");
    static_assert(NUM_Q_HEADS % NUM_KV_HEADS == 0, "Q heads must be divisible by KV heads.");
    static_assert(PAGE_SIZE % KV_BLOCK_SIZE == 0, "PAGE_SIZE must be divisible by KV_BLOCK_SIZE.");

    static constexpr int GQA_RATIO = NUM_Q_HEADS / NUM_KV_HEADS;
    static constexpr int ITERS_PER_PAGE = PAGE_SIZE / KV_BLOCK_SIZE;
    static constexpr int NUM_STAGES = 3;

    using q_rt = kittens::rt_bf<16, HEAD_DIM>;  // only GQA_RATIO rows are used
    using k_rt = kittens::rt_bf<KV_BLOCK_SIZE, HEAD_DIM>;
    using v_rt = kittens::rt_bf<KV_BLOCK_SIZE, HEAD_DIM, kittens::ducks::rt_layout::col>;
    using kv_st = kittens::st_bf<KV_BLOCK_SIZE, HEAD_DIM>;
    using attn_fl_rt = kittens::rt_fl<16, KV_BLOCK_SIZE>;
    using attn_bf_rt = kittens::rt_bf<16, KV_BLOCK_SIZE>;
    using max_vec_rv = typename kittens::rt_fl<16, HEAD_DIM>::col_vec;
    using norm_vec_rv = typename kittens::rt_fl<16, HEAD_DIM>::col_vec;
    using o_rt = kittens::rt_fl<16, HEAD_DIM>;
    using o_sv_bf = kittens::sv_bf<HEAD_DIM>;

    struct parsed_instruction {
        int base_page;
        int seq_idx;

        __device__ inline parsed_instruction(const instruction_t &instruction) {
            base_page = instruction.indices[0];
            seq_idx   = instruction.indices[1];
        }

        __device__ inline parsed_instruction(state_t<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    __device__ static inline int seq_start_page(const parsed_instruction &inst) {
        return inst.base_page + inst.seq_idx * PAGES_PER_SEQ;
    }

    __device__ static inline int seq_len(const Globals &g) {
        return static_cast<int>(g.template gls<SCALAR_POS_ID>().raw_ptr[0]) + 1;
    }

    __device__ static inline int num_kv_iters(const Globals &g) {
        return (seq_len(g) + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
    }

    __device__ static inline int cache_page(const parsed_instruction &inst, int kv_iter_idx) {
        return seq_start_page(inst) + kv_iter_idx / ITERS_PER_PAGE;
    }

    __device__ static inline int iter_in_page(int kv_iter_idx) {
        return kv_iter_idx % ITERS_PER_PAGE;
    }

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            return query;
        }

        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            return 0;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            // TODO: load K/V blocks for this sequence's contiguous static pages.
        }
    };

    struct launcher {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            // TODO: add Blackwell/TMEM launcher work if needed.
        }
    };

    struct consumer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            // TODO: one consumer warp per KV head, computing its GQA_RATIO Q heads.
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            // TODO: store [NUM_Q_HEADS, HEAD_DIM] output for inst.seq_idx.
        }
    };
};

} // namespace llama70b
} // namespace megakittens
