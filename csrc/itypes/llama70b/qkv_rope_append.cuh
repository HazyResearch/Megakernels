#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "itypes/llama70b/matmul_pipeline.cuh"

namespace megakittens {
namespace llama70b {

template <typename Config, typename Globals,
          int BATCH_SIZE,
          int HIDDEN_DIM, int QKV_DIM, int HEAD_DIM, int PAGE_SIZE,
          int NUM_Q_HEADS, int NUM_KV_HEADS,
          int SRC_X, int SRC_QKV_W, int SRC_ROPE_COS, int SRC_ROPE_SIN,
          int SRC_POS_IDS, int SRC_APPEND_IDS, int SRC_K_CACHE, int SRC_V_CACHE,
          int DST_Q, int DST_K_CACHE = -1, int DST_V_CACHE = -1>
struct QkvRopeAppend {
    static_assert(DST_K_CACHE == -1 || DST_K_CACHE == SRC_K_CACHE);
    static_assert(DST_V_CACHE == -1 || DST_V_CACHE == SRC_V_CACHE);
    static_assert(HEAD_DIM == 128, "QkvRopeAppend70b expects head_dim=128.");
    static_assert(HIDDEN_DIM == NUM_Q_HEADS * HEAD_DIM,
                  "QkvRopeAppend70b expects hidden_dim == num_q_heads * head_dim.");
    static_assert(QKV_DIM == HIDDEN_DIM + 2 * NUM_KV_HEADS * HEAD_DIM,
                  "QkvRopeAppend70b QKV dim mismatch.");

    static constexpr int Mb = 256;
    static constexpr int Nb = 256;
    static constexpr int Kb = 64;
    static constexpr int EPI_PIPE_DEPTH = 8;
    static constexpr int NUM_CONSUMERS = 2;
    static constexpr int M_INST = NUM_CONSUMERS * Mb;
    static constexpr int ROWS_PER_CONSUMER = Mb / 2;
    static constexpr int COLS_PER_CHUNK = Nb / EPI_PIPE_DEPTH;
    static constexpr int Q_DIM = HIDDEN_DIM;
    static constexpr int KV_DIM = NUM_KV_HEADS * HEAD_DIM;

    using out_st_t = kittens::st_bf<ROWS_PER_CONSUMER, COLS_PER_CHUNK>;
    using out_rt_bf_t = kittens::rt_bf<ROWS_PER_CONSUMER / 4, COLS_PER_CHUNK>;
    using head_sv_t = kittens::sv<float, HEAD_DIM>;
    using append_sv_t = kittens::sv<int, M_INST>;

    struct parsed_instruction {
        int layer_idx, base_page, m, n;

        __device__ inline parsed_instruction(const instruction_t &instruction) {
            layer_idx = instruction.indices[0];
            base_page = instruction.indices[1];
            m = instruction.indices[2];
            n = instruction.indices[3];
        }

        __device__ inline parsed_instruction(state_t<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    __device__ static inline int row_tile_idx(const parsed_instruction &pi, int cta_rank, int cid) {
        return (2 * pi.m + cta_rank) * 2 + cid;
    }

    // row-layout rt_bf packs (col 2k, col 2k+1) per bf16x2 reg, so rope is per-register without shuffles.
    template <typename Pipeline>
    __device__ static inline void apply_rope_reg(
            out_rt_bf_t &d_reg,
            const head_sv_t &cos_smem,
            const head_sv_t &sin_smem,
            int global_col_start) {
        const int even_base = 2 * (kittens::laneid() % 4);

        #pragma unroll
        for (int j = 0; j < out_rt_bf_t::width; j++) {
            const int low_e  = (global_col_start + j * 16 + even_base) % HEAD_DIM;
            const int high_e = (global_col_start + j * 16 + even_base + 8) % HEAD_DIM;
            const float cl_e = cos_smem[low_e];
            const float sl_e = sin_smem[low_e];
            const float cl_o = cos_smem[low_e + 1];
            const float sl_o = sin_smem[low_e + 1];
            const float ch_e = cos_smem[high_e];
            const float sh_e = sin_smem[high_e];
            const float ch_o = cos_smem[high_e + 1];
            const float sh_o = sin_smem[high_e + 1];

            #pragma unroll
            for (int i = 0; i < out_rt_bf_t::height; i++) {
                #pragma unroll
                for (int k = 0; k < 2; k++) {
                    auto &r = d_reg.tiles[i][j].data[k];
                    const float xe = __bfloat162float(r.x);
                    const float xo = __bfloat162float(r.y);
                    r.x = __float2bfloat16_rn(xe * cl_e - xo * sl_e);
                    r.y = __float2bfloat16_rn(xo * cl_o + xe * sl_o);
                }
                #pragma unroll
                for (int k = 2; k < 4; k++) {
                    auto &r = d_reg.tiles[i][j].data[k];
                    const float xe = __bfloat162float(r.x);
                    const float xo = __bfloat162float(r.y);
                    r.x = __float2bfloat16_rn(xe * ch_e - xo * sh_e);
                    r.y = __float2bfloat16_rn(xo * ch_o + xe * sh_o);
                }
            }
        }
    }

    template <typename Pipeline>
    __device__ static inline void scatter_kv_tile(
            const Globals &g,
            const parsed_instruction &pi,
            out_st_t &tile,
            const append_sv_t &append_smem,
            int global_col_start,
            bool is_k) {
        auto &kv_gl = is_k ? g.template gls<SRC_K_CACHE>() : g.template gls<SRC_V_CACHE>();

        const int cta_rank = kittens::cluster_ctarank();
        const int cid = kittens::warpgroup::groupid();
        const int row_base = row_tile_idx(pi, cta_rank, cid) * ROWS_PER_CONSUMER;
        const int wg_tid = kittens::warpgroup::warpid() * kittens::WARP_THREADS + kittens::laneid();
        const int kv_col_start = global_col_start - (is_k ? Q_DIM : Q_DIM + KV_DIM);
        const int head_idx = kv_col_start / HEAD_DIM;
        const int dim_start = kv_col_start % HEAD_DIM;

        constexpr int VEC = 8;
        for (int elem = wg_tid; elem < ROWS_PER_CONSUMER * (COLS_PER_CHUNK / VEC);
             elem += kittens::WARPGROUP_WARPS * kittens::WARP_THREADS) {
            const int row = elem / (COLS_PER_CHUNK / VEC);
            const int col = (elem % (COLS_PER_CHUNK / VEC)) * VEC;
            const int local_row = row_base - pi.m * M_INST + row;
            const int append_idx = append_smem[local_row];
            const int page = append_idx / PAGE_SIZE;
            const int offset = append_idx % PAGE_SIZE;
            const int cache_page = pi.base_page + page;
            uint4 v;
            __nv_bfloat162 p0, p1, p2, p3;
            p0.x = tile[{row, col}];     p0.y = tile[{row, col + 1}];
            p1.x = tile[{row, col + 2}]; p1.y = tile[{row, col + 3}];
            p2.x = tile[{row, col + 4}]; p2.y = tile[{row, col + 5}];
            p3.x = tile[{row, col + 6}]; p3.y = tile[{row, col + 7}];
            v.x = *reinterpret_cast<uint32_t *>(&p0);
            v.y = *reinterpret_cast<uint32_t *>(&p1);
            v.z = *reinterpret_cast<uint32_t *>(&p2);
            v.w = *reinterpret_cast<uint32_t *>(&p3);
            *reinterpret_cast<uint4 *>(
                &kv_gl[{cache_page, offset, head_idx, dim_start + col}]) = v;
        }
    }

    struct pipeline_specifics {
        template <typename Pipeline>
        __device__ static inline void consumer_loop(const Globals &g, state_t<Config> &s) {
            parsed_instruction pi{s};
            const int cta_rank = kittens::cluster_ctarank();
            const int cid = kittens::warpgroup::groupid();
            using consumer_group = kittens::group<kittens::WARPGROUP_WARPS * Pipeline::NUM_CONSUMERS>;

            auto &q_gl = g.template gls<DST_Q>();

            typename Pipeline::d_tt_t d_tt = s.tensor_alloc.template allocate<typename Pipeline::d_tt_t>(cid * Nb);
            kittens::wait(Pipeline::outputs_arrived(s), 0);

            if (consumer_group::elect_leader()) all_reuse_barrier_wait<Config>(g, s.instruction());
            consumer_group::sync(4);

            uint8_t *scratch_b0 = static_cast<uint8_t *>(
                s.pages[s.lid_to_pid(Pipeline::B_LIDS[0])].ptr(0));
            head_sv_t &cos_smem = *reinterpret_cast<head_sv_t *>(scratch_b0);
            head_sv_t &sin_smem = *reinterpret_cast<head_sv_t *>(scratch_b0 + sizeof(head_sv_t));
            append_sv_t &append_smem = *reinterpret_cast<append_sv_t *>(
                s.pages[s.lid_to_pid(Pipeline::B_LIDS[1])].ptr(0));
            const int pos_id = static_cast<int>(g.template gls<SRC_POS_IDS>().raw_ptr[0]);
            if (kittens::warpid() % kittens::WARPGROUP_WARPS == 0) {
                kittens::warp::load(cos_smem, g.template gls<SRC_ROPE_COS>(), {pos_id, 0});
                kittens::warp::load(sin_smem, g.template gls<SRC_ROPE_SIN>(), {pos_id, 0});
                kittens::warp::load(append_smem, g.template gls<SRC_APPEND_IDS>(), {pi.m});
            }
            consumer_group::sync(4);

            #pragma unroll
            for (int i = 0; i < EPI_PIPE_DEPTH; i++) {
                const int slot = i % Pipeline::NUM_D_TILES;
                const int global_chunk = EPI_PIPE_DEPTH * pi.n + i;
                const int global_col_start = global_chunk * COLS_PER_CHUNK;

                out_rt_bf_t d_reg;
                kittens::warpgroup::load_async(
                    d_reg,
                    d_tt.template subtile<kittens::tt<float, ROWS_PER_CONSUMER, COLS_PER_CHUNK>>(
                        0, COLS_PER_CHUNK * i));
                kittens::tensor_load_wait();

                if (global_col_start < Q_DIM + KV_DIM) {
                    apply_rope_reg<Pipeline>(d_reg, cos_smem, sin_smem, global_col_start);
                }

                kittens::warpgroup::tma::store_async_read_wait<Pipeline::NUM_D_TILES - 1>();
                kittens::warpgroup::sync(cid + 1);
                out_st_t &out_tile = Pipeline::d_st(s, cid, slot);
                kittens::warpgroup::store(out_tile, d_reg);
                kittens::warpgroup::sync(cid + 1);

                if (global_col_start < Q_DIM) {
                    kittens::warpgroup::tma::store_async(
                        q_gl, out_tile,
                        {0, 0, row_tile_idx(pi, cta_rank, cid), global_chunk});
                } else if (global_col_start < Q_DIM + KV_DIM) {
                    scatter_kv_tile<Pipeline>(g, pi, out_tile, append_smem, global_col_start, true);
                    kittens::warpgroup::sync(cid + 1);
                } else {
                    scatter_kv_tile<Pipeline>(g, pi, out_tile, append_smem, global_col_start, false);
                    kittens::warpgroup::sync(cid + 1);
                }
            }

            if (consumer_group::elect_leader()) s.tensor_finish();

            kittens::warpgroup::tma::store_async_wait();
            consumer_group::sync(4);
            if (consumer_group::elect_leader()) {
                __threadfence();
                s.page_finish(s.lid_to_pid(Pipeline::A_LIDS[0]));
                s.page_finish(s.lid_to_pid(Pipeline::B_LIDS[0]));
                s.page_finish(s.lid_to_pid(Pipeline::B_LIDS[1]));
                all_barrier_arrive<Config>(g, s.instruction());
            }
        }
    };

    using pipeline = matmul_pipeline<Config, Globals, BATCH_SIZE, QKV_DIM, HIDDEN_DIM,
                                     Mb, Nb, Kb, EPI_PIPE_DEPTH,
                                     parsed_instruction, pipeline_specifics,
                                     SRC_X, SRC_QKV_W>;

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            return pipeline::lid_release_order(g, s, query);
        }

        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            return pipeline::init_semaphores(g, s);
        }
    };

    struct loader {
        // b_lids are reused as cos/sin/append smem scratch and finished by the consumer.
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            parsed_instruction pi{s};
            const int cta_rank = kittens::cluster_ctarank();
            auto &a_gl = g.template gls<SRC_X>();
            auto &b_gl = g.template gls<SRC_QKV_W>();
            const int num_iters = a_gl.cols() / Kb;

            if (kittens::warp::elect_leader()) {
                all_input_barrier_wait<Config>(g, s.instruction());
                for (int i = 0; i < num_iters + pipeline::LOAD_PIPE_DEPTH; i++) {
                    const int stage = i % pipeline::LOAD_PIPE_DEPTH;
                    if (i < pipeline::LOAD_PIPE_DEPTH) {
                        s.page_wait(s.lid_to_pid(pipeline::A_LIDS[stage]));
                        if (stage % 2 == 0) s.page_wait(s.lid_to_pid(pipeline::B_LIDS[stage / 2]));
                    } else {
                        kittens::wait(pipeline::inputs_finished(s, stage),
                                      ((i + pipeline::LOAD_PIPE_DEPTH) / pipeline::LOAD_PIPE_DEPTH) & 0b1);
                    }
                    if (i < num_iters) {
                        #pragma unroll
                        for (int cid = 0; cid < pipeline::NUM_CONSUMERS; cid++) {
                            kittens::tma::cluster::load_async(
                                pipeline::a_st(s, stage, cid), a_gl,
                                {0, 0, (2 * pi.m + cta_rank) * pipeline::NUM_CONSUMERS + cid, i},
                                pipeline::inputs_arrived(s, stage), (uint16_t)(1 << cta_rank), 0);
                        }
                        kittens::tma::cluster::load_async(
                            pipeline::b_st(s, stage), b_gl,
                            {0, pi.layer_idx, 2 * pi.n + cta_rank, i},
                            pipeline::inputs_arrived(s, stage), (uint16_t)(1 << cta_rank), 0);
                    } else {
                        if (stage != 0) s.page_finish(s.lid_to_pid(pipeline::A_LIDS[stage]));
                    }
                }
            } else if (kittens::warp::elect_leader_from_active()) {
                #pragma unroll
                for (int i = pipeline::NUM_USED_PAGES; i < Config::NUM_PAGES; i++) {
                    s.page_wait(s.lid_to_pid(i));
                    s.page_finish(s.lid_to_pid(i));
                }
            }
        }
    };

    struct launcher {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::launcher_run(g, s);
        }
    };

    struct consumer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::consumer_loop(g, s);
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {}
    };
};

} // namespace llama70b
} // namespace megakittens
