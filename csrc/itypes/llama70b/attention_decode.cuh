#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "utils.cuh"

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
    static constexpr int NUM_CONSUMER_WARPS = NUM_KV_HEADS;
    static constexpr int QO_LID = 0;
    static constexpr int K_LIDS[NUM_STAGES] = {1, 3, 5};
    static constexpr int V_LIDS[NUM_STAGES] = {2, 4, 6};

    using q_rt = kittens::rt_bf<16, HEAD_DIM>;  // only GQA_RATIO rows are used
    using q_st = kittens::st_bf<16, HEAD_DIM, false>;  // Q is loaded as a flat vector
    using k_rt = kittens::rt_bf<KV_BLOCK_SIZE, HEAD_DIM>;
    using v_rt = kittens::rt_bf<KV_BLOCK_SIZE, HEAD_DIM, kittens::ducks::rt_layout::col>;
    using kv_st = kittens::st_bf<KV_BLOCK_SIZE, HEAD_DIM>;
    using attn_fl_rt = kittens::rt_fl<16, KV_BLOCK_SIZE>;
    using attn_bf_rt = kittens::rt_bf<16, KV_BLOCK_SIZE>;
    using max_vec_rv = typename kittens::rt_fl<16, HEAD_DIM>::col_vec;
    using norm_vec_rv = typename kittens::rt_fl<16, HEAD_DIM>::col_vec;
    using o_rt = kittens::rt_fl<16, HEAD_DIM>;
    using o_sv_bf = kittens::sv_bf<HEAD_DIM>;
    using q_row_sv = kittens::sv_bf<NUM_Q_HEADS * HEAD_DIM>;
    using o_full_sv = kittens::sv_bf<NUM_Q_HEADS * HEAD_DIM>;

    static constexpr int SEM_Q_ARRIVED = 0;
    static constexpr int SEM_O_ARRIVED = 1;
    static constexpr int SEM_K_ARRIVED = 2;
    static constexpr int SEM_V_ARRIVED = SEM_K_ARRIVED + NUM_STAGES;
    static constexpr int SEM_K_FINISHED = SEM_V_ARRIVED + NUM_STAGES;
    static constexpr int SEM_V_FINISHED = SEM_K_FINISHED + NUM_STAGES;
    static constexpr int SEM_COUNT = SEM_V_FINISHED + NUM_STAGES;

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

    __device__ static inline int qo_pid(state_t<Config> &s) {
        return s.lid_to_pid(QO_LID);
    }

    __device__ static inline int k_pid(state_t<Config> &s, int stage) {
        return s.lid_to_pid(K_LIDS[stage]);
    }

    __device__ static inline int v_pid(state_t<Config> &s, int stage) {
        return s.lid_to_pid(V_LIDS[stage]);
    }

    __device__ static inline q_row_sv &q_smem(state_t<Config> &s) {
        return s.pages[qo_pid(s)].template as<q_row_sv>();
    }

    __device__ static inline q_st &q_group_smem(state_t<Config> &s, int kv_head_idx) {
        return s.pages[qo_pid(s)].template as<q_st>(
            kv_head_idx * GQA_RATIO * sizeof(o_sv_bf));
    }

    __device__ static inline o_sv_bf &o_smem(state_t<Config> &s, int q_head_idx) {
        return s.pages[qo_pid(s)].template as<o_sv_bf>(
            sizeof(q_row_sv) + q_head_idx * sizeof(o_sv_bf));
    }

    __device__ static inline o_full_sv &o_full_smem(state_t<Config> &s) {
        return s.pages[qo_pid(s)].template as<o_full_sv>(sizeof(q_row_sv));
    }

    __device__ static inline kv_st &k_smem(state_t<Config> &s, int stage, int kv_head_idx) {
        return s.pages[k_pid(s, stage)].template as<kv_st>(kv_head_idx * sizeof(kv_st));
    }

    __device__ static inline kv_st &v_smem(state_t<Config> &s, int stage, int kv_head_idx) {
        return s.pages[v_pid(s, stage)].template as<kv_st>(kv_head_idx * sizeof(kv_st));
    }

    __device__ static inline kittens::semaphore &Q_arrived(state_t<Config> &s) {
        return s.semaphores()[SEM_Q_ARRIVED];
    }

    __device__ static inline kittens::semaphore &O_arrived(state_t<Config> &s) {
        return s.semaphores()[SEM_O_ARRIVED];
    }

    __device__ static inline kittens::semaphore &K_arrived(state_t<Config> &s, int stage) {
        return s.semaphores()[SEM_K_ARRIVED + stage];
    }

    __device__ static inline kittens::semaphore &V_arrived(state_t<Config> &s, int stage) {
        return s.semaphores()[SEM_V_ARRIVED + stage];
    }

    __device__ static inline kittens::semaphore &K_finished(state_t<Config> &s, int stage) {
        return s.semaphores()[SEM_K_FINISHED + stage];
    }

    __device__ static inline kittens::semaphore &V_finished(state_t<Config> &s, int stage) {
        return s.semaphores()[SEM_V_FINISHED + stage];
    }

    template <kittens::ducks::rt::row_layout RT>
    __device__ static inline void right_fill(
        RT &dst, const RT &src, int col_idx,
        typename kittens::base_types::packing<typename RT::dtype>::unpacked_type val = 0) {
        if (col_idx >= dst.cols) return;
        #pragma unroll
        for (int i = 0; i < dst.height; i++) {
            #pragma unroll
            for (int j = 0; j < dst.width; j++) {
                #pragma unroll
                for (int k = 0; k < dst.packed_per_tile; k++) {
                    int col_idx_x = (j * dst.tile_size_col) + ((k / 2) * 8) + ((kittens::laneid() % 4) * 2);
                    int col_idx_y = col_idx_x + 1;
                    dst.tiles[i][j].data[k].x = (col_idx_x >= col_idx) ? val : src.tiles[i][j].data[k].x;
                    dst.tiles[i][j].data[k].y = (col_idx_y >= col_idx) ? val : src.tiles[i][j].data[k].y;
                }
            }
        }
    }

    template <kittens::ducks::sv::all SV, kittens::ducks::rt::all RT>
    __device__ static inline void store_8_rows(SV *dst, const RT &src) {
        static_assert(RT::rows == 16, "src rows must be 16.");
        static_assert(SV::length == RT::cols, "dst length must match src cols.");

        using T2 = typename RT::dtype;
        using U = typename SV::dtype;
        using U2 = typename kittens::base_types::packing<U>::packed_type;

        uint32_t dst_ptr[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            dst_ptr[i] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[i].data[0]));
        }

        int lane_id = kittens::laneid();
        int local_row_idx = lane_id / 4;
        int local_col_idx = lane_id % 4;

        if (lane_id < 32) {
            for (int j = 0; j < src.width; j++) {
                U2 tmp[2];
                tmp[0] = kittens::base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[0]);
                tmp[1] = kittens::base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[2]);
                int col_idx = local_col_idx * 2 + j * 16;
                kittens::move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                kittens::move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx + 8), tmp[1]);
            }
        }
    }

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            static_assert(Config::NUM_PAGES == 7 && NUM_STAGES == 3);
            switch (num_kv_iters(g) % NUM_STAGES) {
                case 0: { constexpr int order[] = {1, 2, 3, 4, 5, 6, 0}; return order[query]; }
                case 1: { constexpr int order[] = {3, 4, 5, 6, 1, 2, 0}; return order[query]; }
                case 2: { constexpr int order[] = {5, 6, 1, 2, 3, 4, 0}; return order[query]; }
            }
            return 0;
        }

        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            const int lane_id = kittens::laneid();
            if (lane_id == SEM_Q_ARRIVED) {
                kittens::init_semaphore(Q_arrived(s), 1);
            } else if (lane_id == SEM_O_ARRIVED) {
                kittens::init_semaphore(O_arrived(s), NUM_CONSUMER_WARPS);
            } else if (lane_id >= SEM_K_ARRIVED && lane_id < SEM_V_ARRIVED) {
                kittens::init_semaphore(K_arrived(s, lane_id - SEM_K_ARRIVED), 1);
            } else if (lane_id >= SEM_V_ARRIVED && lane_id < SEM_K_FINISHED) {
                kittens::init_semaphore(V_arrived(s, lane_id - SEM_V_ARRIVED), 1);
            } else if (lane_id >= SEM_K_FINISHED && lane_id < SEM_V_FINISHED) {
                kittens::init_semaphore(K_finished(s, lane_id - SEM_K_FINISHED), NUM_CONSUMER_WARPS);
            } else if (lane_id >= SEM_V_FINISHED && lane_id < SEM_COUNT) {
                kittens::init_semaphore(V_finished(s, lane_id - SEM_V_FINISHED), NUM_CONSUMER_WARPS);
            }
            return SEM_COUNT;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            if (kittens::warp::elect_leader()) {
                parsed_instruction inst{s};
                all_input_barrier_wait<Config>(g, s.instruction());

                auto &q_gl = g.template gls<SRC_Q>();
                auto &k_gl = g.template gls<SRC_K_CACHE>();
                auto &v_gl = g.template gls<SRC_V_CACHE>();
                const int total_kv_iters = num_kv_iters(g);

                s.page_wait(qo_pid(s));
                kittens::tma::expect_bytes(Q_arrived(s), sizeof(q_row_sv));
                kittens::tma::load_async<kittens::cache_policy::EVICT_LAST>(
                    q_smem(s), q_gl, {0, 0, inst.seq_idx, 0}, Q_arrived(s));

                for (int kv_iter_idx = 0; kv_iter_idx < total_kv_iters; kv_iter_idx++) {
                    const int stage = kv_iter_idx % NUM_STAGES;
                    const int page_idx = cache_page(inst, kv_iter_idx);
                    const int page_iter = iter_in_page(kv_iter_idx);

                    if (kv_iter_idx < NUM_STAGES) {
                        s.page_wait(k_pid(s, stage));
                        s.page_wait(v_pid(s, stage));
                    } else {
                        kittens::wait(K_finished(s, stage), (kv_iter_idx / NUM_STAGES - 1) & 1);
                        kittens::wait(V_finished(s, stage), (kv_iter_idx / NUM_STAGES - 1) & 1);
                    }

                    kittens::tma::expect_bytes(K_arrived(s, stage), NUM_KV_HEADS * sizeof(kv_st));
                    kittens::tma::expect_bytes(V_arrived(s, stage), NUM_KV_HEADS * sizeof(kv_st));
                    #pragma unroll
                    for (int kv_head_idx = 0; kv_head_idx < NUM_KV_HEADS; kv_head_idx++) {
                        kittens::tma::load_async<kittens::dim::DEPTH, kittens::cache_policy::EVICT_FIRST>(
                            k_smem(s, stage, kv_head_idx), k_gl,
                            {page_idx, page_iter, kv_head_idx, 0},
                            K_arrived(s, stage));
                        kittens::tma::load_async<kittens::dim::DEPTH, kittens::cache_policy::EVICT_FIRST>(
                            v_smem(s, stage, kv_head_idx), v_gl,
                            {page_idx, page_iter, kv_head_idx, 0},
                            V_arrived(s, stage));
                    }
                }

                for (int stage = total_kv_iters; stage < NUM_STAGES; stage++) {
                    s.page_wait(k_pid(s, stage));
                    s.page_wait(v_pid(s, stage));
                    s.page_finish(k_pid(s, stage));
                    s.page_finish(v_pid(s, stage));
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
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            static_assert(NUM_CONSUMER_WARPS == NUM_KV_HEADS);
            using consumer_group = kittens::group<NUM_CONSUMER_WARPS>;
            const int kv_head_idx = kittens::warpid();
            if (kv_head_idx >= NUM_KV_HEADS) return;

            const int total_kv_iters = num_kv_iters(g);
            const int valid_seq_len = seq_len(g);
            const float softmax_temp =
                g.template gls<SCALAR_ATTN_SCALE>().raw_ptr[0] * 1.44269504089f;

            q_rt Q_reg;
            k_rt K_reg;
            v_rt V_reg;
            o_rt O_reg;
            attn_fl_rt attn_fl_reg;
            attn_bf_rt attn_bf_reg;
            max_vec_rv scaled_max_vec_reg;
            max_vec_rv last_scaled_max_vec_reg;
            max_vec_rv diff_scaled_max_vec_reg;
            norm_vec_rv norm_vec_reg;

            kittens::warp::neg_infty(scaled_max_vec_reg);
            kittens::warp::neg_infty(last_scaled_max_vec_reg);
            kittens::warp::zero(norm_vec_reg);
            kittens::warp::zero(O_reg);

            kittens::wait(Q_arrived(s), 0);
            kittens::warp::load(Q_reg, q_group_smem(s, kv_head_idx));

            for (int kv_iter_idx = 0; kv_iter_idx < total_kv_iters; kv_iter_idx++) {
                const int stage = kv_iter_idx % NUM_STAGES;

                kittens::warp::zero(attn_fl_reg);
                kittens::wait(K_arrived(s, stage), (kv_iter_idx / NUM_STAGES) & 1);
                kittens::warp::load(K_reg, k_smem(s, stage, kv_head_idx));
                kittens::warp::mma_ABt(attn_fl_reg, Q_reg, K_reg, attn_fl_reg);
                kittens::warp::sync();
                kittens::warp::arrive(K_finished(s, stage));

                if ((kv_iter_idx + 1) * KV_BLOCK_SIZE > valid_seq_len) {
                    right_fill(
                        attn_fl_reg, attn_fl_reg, valid_seq_len % KV_BLOCK_SIZE,
                        kittens::base_types::constants<float>::neg_infty());
                }

                kittens::warp::mul(attn_fl_reg, attn_fl_reg, softmax_temp);
                kittens::warp::row_max(scaled_max_vec_reg, attn_fl_reg, scaled_max_vec_reg);
                kittens::warp::sub_row(attn_fl_reg, attn_fl_reg, scaled_max_vec_reg);
                kittens::warp::exp2(attn_fl_reg, attn_fl_reg);
                kittens::warp::sub(diff_scaled_max_vec_reg, last_scaled_max_vec_reg, scaled_max_vec_reg);
                kittens::warp::exp2(diff_scaled_max_vec_reg, diff_scaled_max_vec_reg);

                kittens::warp::mul_row(O_reg, O_reg, diff_scaled_max_vec_reg);
                kittens::wait(V_arrived(s, stage), (kv_iter_idx / NUM_STAGES) & 1);
                kittens::warp::load(V_reg, v_smem(s, stage, kv_head_idx));
                kittens::warp::copy(attn_bf_reg, attn_fl_reg);
                kittens::warp::mma_AB(O_reg, attn_bf_reg, V_reg, O_reg);
                kittens::warp::sync();
                kittens::warp::arrive(V_finished(s, stage));

                kittens::warp::mul(norm_vec_reg, norm_vec_reg, diff_scaled_max_vec_reg);
                kittens::warp::row_sum(norm_vec_reg, attn_fl_reg, norm_vec_reg);
                kittens::warp::copy(last_scaled_max_vec_reg, scaled_max_vec_reg);
            }

            consumer_group::sync(1);
            if (consumer_group::elect_leader()) {
                #pragma unroll
                for (int stage = 0; stage < NUM_STAGES; stage++) {
                    if (stage < total_kv_iters) {
                        s.page_finish(k_pid(s, stage));
                        s.page_finish(v_pid(s, stage));
                    }
                }
            }
            consumer_group::sync(1);

            kittens::warp::div_row(O_reg, O_reg, norm_vec_reg);

            const int q_head_start = kv_head_idx * GQA_RATIO;
            store_8_rows(&o_smem(s, q_head_start), O_reg);
            kittens::warp::sync();
            kittens::warp::arrive(O_arrived(s));
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            parsed_instruction inst{s};
            auto &out_gl = g.template gls<DST_OUT>();

            if (kittens::warp::elect_leader()) {
                kittens::wait(O_arrived(s), 0);
                all_reuse_barrier_wait<Config>(g, s.instruction());

                kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                    out_gl, o_full_smem(s),
                    kittens::coord<o_full_sv>{0, 0, inst.seq_idx, 0});
                kittens::tma::store_async_wait();
                s.page_finish(qo_pid(s));
                all_barrier_arrive<Config>(g, s.instruction());
            }
        }
    };
};

} // namespace llama70b
} // namespace megakittens
