#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "utils.cuh"

namespace megakittens {

template <typename Config, typename Globals,
          int HEAD_DIM, int KV_BLOCK_SIZE, int GQA_RATIO,
          int SRC_Q, int SRC_K_CACHE, int SRC_V_CACHE, int DST>
struct AttentionPartial {

    static_assert(GQA_RATIO == 4, "GQA_RATIO must be 4.");
    static constexpr int NUM_STAGES = 3;
    static constexpr int QOL_PAGE = 0;
    static constexpr int KV_PAGE = 1;

    using q_st = kittens::st_bf<16, HEAD_DIM>;
    using q_rt = kittens::rt_bf<16, HEAD_DIM>;
    using k_rt = kittens::rt_bf<KV_BLOCK_SIZE, HEAD_DIM>;
    using v_rt = kittens::rt_bf<KV_BLOCK_SIZE, HEAD_DIM, kittens::ducks::rt_layout::col>;
    using kv_st = kittens::st_bf<KV_BLOCK_SIZE, HEAD_DIM>;
    using attn_fl_rt = kittens::rt_fl<16, KV_BLOCK_SIZE>;
    using attn_bf_rt = kittens::rt_bf<16, KV_BLOCK_SIZE>;
    using max_vec_rv = typename kittens::rt_fl<16, HEAD_DIM>::col_vec;
    using norm_vec_rv = typename kittens::rt_fl<16, HEAD_DIM>::col_vec;
    using l_rv = typename kittens::rt_fl<16, HEAD_DIM>::col_vec;
    using o_rt = kittens::rt_fl<16, HEAD_DIM>;
    using o_sv = kittens::sv_fl<HEAD_DIM>;
    using o_sv_bf = kittens::sv_bf<HEAD_DIM>;
    using l_sv = kittens::sv_fl<16>;

    struct parsed_instruction {
        int layer_idx;
        int kv_head_idx;
        int barrier_base;
        __device__ inline parsed_instruction(const instruction_t &instruction) {
            layer_idx    = instruction.indices[0];
            kv_head_idx  = instruction.indices[1];
            barrier_base = instruction.indices[2];
        }
        __device__ inline parsed_instruction(state_t<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    __device__ static inline kittens::semaphore &Q_arrived(state_t<Config> &s) {
        return s.semaphores()[0];
    }
    __device__ static inline kittens::semaphore &O_arrived(state_t<Config> &s) {
        return s.semaphores()[1];
    }
    __device__ static inline kittens::semaphore &L_arrived(state_t<Config> &s) {
        return s.semaphores()[2];
    }
    __device__ static inline kittens::semaphore &K_arrived(state_t<Config> &s, int stage) {
        return s.semaphores()[3 + stage * 2];
    }
    __device__ static inline kittens::semaphore &V_arrived(state_t<Config> &s, int stage) {
        return s.semaphores()[3 + stage * 2 + 1];
    }
    __device__ static inline kittens::semaphore &K_finished(state_t<Config> &s, int stage) {
        return s.semaphores()[3 + NUM_STAGES * 2 + stage * 2];
    }
    __device__ static inline kittens::semaphore &V_finished(state_t<Config> &s, int stage) {
        return s.semaphores()[3 + NUM_STAGES * 2 + stage * 2 + 1];
    }
    static constexpr int SEM_COUNT = 3 + 4 * NUM_STAGES;

    __device__ static inline int qol_pid(state_t<Config> &s) { return s.lid_to_pid(QOL_PAGE); }
    __device__ static inline int kv_pid(state_t<Config> &s)  { return s.lid_to_pid(KV_PAGE); }

    __device__ static inline q_st &get_Q_smem(state_t<Config> &s) {
        return s.pages[qol_pid(s)].template as<q_st>();
    }
    __device__ static inline o_sv (&get_O_smem(state_t<Config> &s))[4] {
        return *reinterpret_cast<o_sv(*)[4]>(
            reinterpret_cast<char *>(s.pages[qol_pid(s)].ptr(sizeof(q_st))));
    }
    __device__ static inline l_sv &get_L_smem(state_t<Config> &s) {
        return *reinterpret_cast<l_sv *>(
            reinterpret_cast<char *>(s.pages[qol_pid(s)].ptr(sizeof(q_st) + sizeof(o_sv) * 4)));
    }
    __device__ static inline kv_st &get_K_smem(state_t<Config> &s, int stage) {
        return *reinterpret_cast<kv_st *>(
            reinterpret_cast<char *>(s.pages[kv_pid(s)].ptr(sizeof(kv_st) * (stage * 2))));
    }
    __device__ static inline kv_st &get_V_smem(state_t<Config> &s, int stage) {
        return *reinterpret_cast<kv_st *>(
            reinterpret_cast<char *>(s.pages[kv_pid(s)].ptr(sizeof(kv_st) * (1 + stage * 2))));
    }

    __device__ static inline void
    load_Q_async(q_st &dst, const Globals &g, int q_head_start_idx) {
        static_assert(HEAD_DIM == 64 && GQA_RATIO == 4, "Fix this function.");
        constexpr int elem_per_memcpy = sizeof(float4) / sizeof(kittens::bf16); // 8
        constexpr int memcpy_per_row = HEAD_DIM / elem_per_memcpy;              // 8

        const kittens::bf16 *src_ptr = reinterpret_cast<const kittens::bf16 *>(
            g.template gls<SRC_Q>().raw_ptr) + q_head_start_idx * HEAD_DIM;
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(
            &dst.data[(q_head_start_idx % 16) * HEAD_DIM]));

        int laneid = kittens::laneid();
        int row = laneid / memcpy_per_row;
        int col = (laneid * elem_per_memcpy) % HEAD_DIM;

        asm volatile(
            "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(
                dst.idx(dst_ptr, {row, col})),
            "l"(&src_ptr[row * HEAD_DIM + col])
            : "memory");
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    template <kittens::ducks::sv::all SV, kittens::ducks::rt::all RT>
    __device__ static inline void
    store_4_rows(SV (&dst)[4], const RT &src, int row4idx) {
        static_assert(RT::rows == 16, "src rows must be 16.");
        static_assert(SV::length == RT::cols, "dst length must match src cols.");

        using T2 = typename RT::dtype;
        using T = typename kittens::base_types::packing<T2>::unpacked_type;
        using U = typename SV::dtype;
        using U2 = typename kittens::base_types::packing<U>::packed_type;

        uint32_t dst_ptr[4];
        dst_ptr[0] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[0].data[0]));
        dst_ptr[1] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[1].data[0]));
        dst_ptr[2] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[2].data[0]));
        dst_ptr[3] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[3].data[0]));

        int laneid = kittens::laneid();
        int local_row_idx = (laneid % 16) / 4;
        int local_col_idx = laneid % 4;

        if (row4idx % 2 == 0 && laneid < 16) {
            int data_idx = (row4idx / 2 == 0) ? 0 : 1;
            int data_idx2 = (row4idx / 2 == 0) ? 2 : 3;
            for (int j = 0; j < src.width; j++) {
                U2 tmp[2];
                tmp[0] = kittens::base_types::convertor<U2, T2>::convert(
                    src.tiles[0][j].data[data_idx]);
                tmp[1] = kittens::base_types::convertor<U2, T2>::convert(
                    src.tiles[0][j].data[data_idx2]);
                int col_idx = local_col_idx * 2 + j * 16;
                kittens::move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                kittens::move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx + 8), tmp[1]);
            }
        } else if (row4idx % 2 == 1 && laneid >= 16) {
            int data_idx = (row4idx / 2 == 0) ? 0 : 1;
            int data_idx2 = (row4idx / 2 == 0) ? 2 : 3;
            for (int j = 0; j < src.width; j++) {
                U2 tmp[2];
                tmp[0] = kittens::base_types::convertor<U2, T2>::convert(
                    src.tiles[0][j].data[data_idx]);
                tmp[1] = kittens::base_types::convertor<U2, T2>::convert(
                    src.tiles[0][j].data[data_idx2]);
                int col_idx = local_col_idx * 2 + j * 16;
                kittens::move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                kittens::move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx + 8), tmp[1]);
            }
        }
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
                    int col_idx_x = (j * dst.tile_size_col) + ((k / 2) * 8) +
                                    ((kittens::laneid() % 4) * 2);
                    int col_idx_y = col_idx_x + 1;
                    dst.tiles[i][j].data[k].x = (col_idx_x >= col_idx) ? val : src.tiles[i][j].data[k].x;
                    dst.tiles[i][j].data[k].y = (col_idx_y >= col_idx) ? val : src.tiles[i][j].data[k].y;
                }
            }
        }
    }

    struct controller {
        __device__ __forceinline__ static int
        lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            if (query < Config::NUM_PAGES - 2)
                return query + 2;
            return query - (Config::NUM_PAGES - 2);
        }
        __device__ __forceinline__ static int
        init_semaphores(const Globals &g, state_t<Config> &s) {
            if (kittens::laneid() == 0)
                kittens::init_semaphore(Q_arrived(s), 1);
            if (kittens::laneid() == 1)
                kittens::init_semaphore(O_arrived(s), 1);
            if (kittens::laneid() == 2)
                kittens::init_semaphore(L_arrived(s), 1);
            if (kittens::laneid() < NUM_STAGES)
                kittens::init_semaphore(K_arrived(s, kittens::laneid()), 1);
            if (kittens::laneid() < NUM_STAGES)
                kittens::init_semaphore(V_arrived(s, kittens::laneid()), 1);
            if (kittens::laneid() < NUM_STAGES)
                kittens::init_semaphore(K_finished(s, kittens::laneid()), 1);
            if (kittens::laneid() < NUM_STAGES)
                kittens::init_semaphore(V_finished(s, kittens::laneid()), 1);
            return SEM_COUNT;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            int laneid = kittens::laneid();
            if (laneid >= 2 && laneid < Config::NUM_PAGES) {
                int pid = s.lid_to_pid(laneid);
                s.page_wait(pid);
                s.page_finish(pid);
            }
        }
    };

    struct launcher {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            s.tensor_wait();
            if (kittens::warp::elect_leader()) s.tensor_finish();

            if (kittens::warp::elect_leader()) {

                parsed_instruction inst{s};
                int seq_len = g.pos_id + 1;
                int total_attn_blocks = (seq_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;

                s.page_wait(kv_pid(s));

                if (total_attn_blocks == 0) {
                    s.page_finish(kv_pid(s));
                }

                for (int i = 0; i < total_attn_blocks; i++) {
                    int stage = i % NUM_STAGES;
                    kv_st &K_smem = get_K_smem(s, stage);
                    kv_st &V_smem = get_V_smem(s, stage);

                    if (i == total_attn_blocks - 1) {
                        all_reuse_barrier_wait<Config>(g, s.instruction());
                    }

                    if (i >= NUM_STAGES) {
                        kittens::wait(K_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                        kittens::wait(V_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                    }

                    kittens::tma::expect(K_arrived(s, stage), K_smem);
                    kittens::tma::load_async<kittens::dim::DEPTH, kittens::cache_policy::EVICT_FIRST>(
                        K_smem, g.template gls<SRC_K_CACHE>(),
                        {inst.layer_idx, i, inst.kv_head_idx, 0},
                        K_arrived(s, stage));
                    kittens::tma::expect(V_arrived(s, stage), V_smem);
                    kittens::tma::load_async<kittens::dim::DEPTH, kittens::cache_policy::EVICT_FIRST>(
                        V_smem, g.template gls<SRC_V_CACHE>(),
                        {inst.layer_idx, i, inst.kv_head_idx, 0},
                        V_arrived(s, stage));
                }
            }
        }
    };

    struct consumer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            if (kittens::warpid() != 0) return;

            parsed_instruction inst{s};
            int q_head_start_idx = inst.kv_head_idx * GQA_RATIO;
            int q_head_local_idx = (q_head_start_idx % 16) / 4;
            int seq_len = g.pos_id + 1;
            int total_attn_blocks = (seq_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
            float softmax_temp = g.attn_scale * 1.44269504089f; // 1 / (sqrt(D_h) * ln(2))

            q_rt Q_reg;
            k_rt K_reg;
            v_rt V_reg;
            l_rv L_reg;
            o_rt O_reg;
            attn_fl_rt attn_fl_reg;
            attn_bf_rt attn_bf_reg;
            max_vec_rv max_vec_reg;
            max_vec_rv scaled_max_vec_reg;
            max_vec_rv last_scaled_max_vec_reg;
            max_vec_rv diff_scaled_max_vec_reg;
            norm_vec_rv norm_vec_reg;
            kittens::warp::neg_infty(max_vec_reg);
            kittens::warp::zero(last_scaled_max_vec_reg);
            kittens::warp::zero(norm_vec_reg);
            kittens::warp::zero(O_reg);
            o_sv (&O_smem)[4] = get_O_smem(s);
            l_sv &L_smem = get_L_smem(s);

            if (kittens::warp::elect_leader()) {
                s.record(TEVENT_AT_GMEM_WAIT);
                all_input_barrier_wait<Config>(g, s.instruction());
                s.record(TEVENT_DONE_GMEM_WAIT);
            }
            kittens::warp::sync();

            s.record(TEVENT_CONSUMER_START);
            s.page_wait(qol_pid(s));
            q_st &Q_smem = get_Q_smem(s);
            load_Q_async(Q_smem, g, q_head_start_idx);
            kittens::warp::load_async_wait();
            kittens::warp::load(Q_reg, Q_smem);

            for (int i = 0; i < total_attn_blocks; i++) {
                int stage = i % NUM_STAGES;
                kv_st &K_smem_tile = get_K_smem(s, stage);
                kv_st &V_smem_tile = get_V_smem(s, stage);

                kittens::warp::zero(attn_fl_reg);
                kittens::warp::wait(K_arrived(s, stage), (i / NUM_STAGES) % 2);
                kittens::warp::load(K_reg, K_smem_tile);
                kittens::warp::mma_ABt(attn_fl_reg, Q_reg, K_reg, attn_fl_reg);
                kittens::warp::sync();
                kittens::warp::arrive(K_finished(s, stage));

                if ((i + 1) * KV_BLOCK_SIZE > seq_len)
                    right_fill(attn_fl_reg, attn_fl_reg,
                               seq_len % KV_BLOCK_SIZE, -999999999999.f);

                kittens::warp::row_max(max_vec_reg, attn_fl_reg, max_vec_reg);

                kittens::warp::mul(attn_fl_reg, attn_fl_reg, softmax_temp);
                kittens::warp::mul(scaled_max_vec_reg, max_vec_reg, softmax_temp);

                kittens::warp::sub_row(attn_fl_reg, attn_fl_reg, scaled_max_vec_reg);
                kittens::warp::exp2(attn_fl_reg, attn_fl_reg);

                kittens::warp::sub(diff_scaled_max_vec_reg, last_scaled_max_vec_reg, scaled_max_vec_reg);
                kittens::warp::exp2(diff_scaled_max_vec_reg, diff_scaled_max_vec_reg);

                kittens::warp::mul_row(O_reg, O_reg, diff_scaled_max_vec_reg);
                kittens::warp::wait(V_arrived(s, stage), (i / NUM_STAGES) % 2);
                kittens::warp::load(V_reg, V_smem_tile);
                kittens::warp::copy(attn_bf_reg, attn_fl_reg);
                kittens::warp::mma_AB(O_reg, attn_bf_reg, V_reg, O_reg);
                kittens::warp::sync();
                kittens::warp::arrive(V_finished(s, stage));

                kittens::warp::mul(norm_vec_reg, norm_vec_reg, diff_scaled_max_vec_reg);
                kittens::warp::row_sum(norm_vec_reg, attn_fl_reg, norm_vec_reg);

                kittens::warp::copy(last_scaled_max_vec_reg, scaled_max_vec_reg);
            }

            kittens::warp::sync();

            if (total_attn_blocks > 0) {
                if (kittens::warp::elect_leader())
                    s.page_finish(kv_pid(s));
                kittens::warp::div_row(O_reg, O_reg, norm_vec_reg);
                kittens::warp::log2(L_reg, norm_vec_reg);
                kittens::warp::add(L_reg, L_reg, last_scaled_max_vec_reg);
            } else {
                kittens::warp::neg_infty(L_reg);
            }

            s.record(TEVENT_OUTPUT_READY);
            store_4_rows(O_smem, O_reg, q_head_local_idx);
            kittens::warp::sync();
            kittens::warp::arrive(O_arrived(s));

            kittens::warp::store(L_smem, L_reg);
            kittens::warp::sync();
            kittens::warp::arrive(L_arrived(s));
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            parsed_instruction inst{s};
            int q_head_start_idx = inst.kv_head_idx * GQA_RATIO;

            o_sv (&O_smem)[4] = get_O_smem(s);

            if (kittens::warp::elect_leader())
                kittens::wait(O_arrived(s), 0);
            kittens::warp::sync();

            kittens::rv_bf<HEAD_DIM> O_bf;
            for (int head_offset = 0; head_offset < GQA_RATIO; head_offset++) {
                auto &smem_fl = O_smem[head_offset];
                auto &smem_bf = *reinterpret_cast<o_sv_bf *>(&smem_fl);

                kittens::warp::load(O_bf, smem_fl);
                kittens::warp::sync();
                kittens::warp::store(smem_bf, O_bf);
                kittens::warp::sync();
            }

            if (kittens::warp::elect_leader()) {
                for (int head_offset = 0; head_offset < GQA_RATIO; head_offset++) {
                    auto &smem_bf = *reinterpret_cast<o_sv_bf *>(&O_smem[head_offset]);
                    kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                        g.template gls<DST>(), smem_bf,
                        {0, q_head_start_idx + head_offset});
                }
            }

            kittens::warp::sync();
            kittens::tma::store_async_wait();
            if (kittens::warp::elect_leader())
                s.page_finish(qol_pid(s));

            // Signal the attn_red barrier (skip_attn_reduction path — no
            // reduction step, so we signal o_proj's dependency directly).
            // barrier_base points to attn_red slot; add GQA_RATIO for
            // the 4 Q heads this instruction covers.
            if (kittens::warp::elect_leader()) {
                barrier_arrive<Config>(&g.barriers.raw_ptr[inst.barrier_base], GQA_RATIO);
            }
        }
    };
};

} // namespace megakittens
