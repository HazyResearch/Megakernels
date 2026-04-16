#pragma once

#include "kittens.cuh"

namespace megakittens {

template <typename Config, typename Globals, int SRC_Q, int SRC_K, int SRC_V, int DST_O, bool CAUSAL = false>
struct Attention {
    static constexpr int Mb = 128;
    static constexpr int Nb = 128;
    static constexpr int Db = 128;
    static constexpr int NUM_SOFTMAXXERS = 2;
    static constexpr int LOAD_STAGES = 4;

    using q_tile = kittens::st_bf<Mb, Db>;       // 32KB
    using k_tile = kittens::st_bf<Nb/2, Db>;     // 16KB
    using v_tile = kittens::st_bf<Nb, Db/2>;     // 16KB
    using o_tile = kittens::st_bf<Mb, Db>;       // 32KB

    using d_tt_scores = kittens::tt<float, Mb, Nb>;
    using d_tt_scores_bf = kittens::tt<kittens::bf16, Mb, Nb>;
    using d_tt_scores_bf_1q = kittens::tt<kittens::bf16, Mb, Nb/4>;
    using d_tt_outputs = kittens::tt<float, Mb, Db>;
    using v_quarter_tile = kittens::st_bf<Nb/4, Db/2>;

    static constexpr int Q_LIDS[NUM_SOFTMAXXERS] = {0, 1};
    static constexpr int KV_LIDS[2] = {2, 3};
    static constexpr int O_LIDS[NUM_SOFTMAXXERS] = {4, 5};
    static constexpr int NUM_USED_PAGES = 6;

    static constexpr int SEM_Q_ARRIVED = 0;
    static constexpr int SEM_KV_ARRIVED = 2;
    static constexpr int SEM_KV_FINISHED = 6;
    static constexpr int SEM_SCORES_ARRIVED = 10;
    static constexpr int SEM_NORM_ARRIVED = 12;
    static constexpr int SEM_NORM_Q_ARRIVED = 14;
    static constexpr int SEM_TILE_ARRIVED = 20;
    static constexpr int SEM_COUNT = 22;

    __device__ static inline kittens::semaphore &q_arrived(state_t<Config> &s, int qid)          { return s.semaphores()[SEM_Q_ARRIVED + qid]; }
    __device__ static inline kittens::semaphore &kv_arrived(state_t<Config> &s, int stage)       { return s.semaphores()[SEM_KV_ARRIVED + stage]; }
    __device__ static inline kittens::semaphore &kv_finished(state_t<Config> &s, int stage)      { return s.semaphores()[SEM_KV_FINISHED + stage]; }
    __device__ static inline kittens::semaphore &scores_arrived(state_t<Config> &s, int qid)     { return s.semaphores()[SEM_SCORES_ARRIVED + qid]; }
    __device__ static inline kittens::semaphore &norm_scores_arrived(state_t<Config> &s, int qid){ return s.semaphores()[SEM_NORM_ARRIVED + qid]; }
    __device__ static inline kittens::semaphore &norm_scores_quarter_arrived(state_t<Config> &s, int q, int qid) { return s.semaphores()[SEM_NORM_Q_ARRIVED + q*NUM_SOFTMAXXERS + qid]; }
    __device__ static inline kittens::semaphore &tile_arrived(state_t<Config> &s, int qid)       { return s.semaphores()[SEM_TILE_ARRIVED + qid]; }

    __device__ static inline q_tile &q_st(state_t<Config> &s, int qid) {
        return s.pages[s.lid_to_pid(Q_LIDS[qid])].template as<q_tile>();
    }
    __device__ static inline k_tile &kv_st(state_t<Config> &s, int stage) {
        return s.pages[s.lid_to_pid(KV_LIDS[stage/2])].template as<k_tile>((stage%2) * sizeof(k_tile));
    }
    __device__ static inline o_tile &o_st(state_t<Config> &s, int qid) {
        return s.pages[s.lid_to_pid(O_LIDS[qid])].template as<o_tile>();
    }

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            return query;
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            const int lane = kittens::laneid();
            if (lane < SEM_COUNT) {
                if (lane < SEM_Q_ARRIVED + NUM_SOFTMAXXERS) {
                    kittens::init_semaphore(s.semaphores()[lane], 1);
                } else if (lane < SEM_KV_ARRIVED + LOAD_STAGES) {
                    kittens::init_semaphore(s.semaphores()[lane], 1);
                } else if (lane < SEM_KV_FINISHED + LOAD_STAGES) {
                    kittens::init_semaphore(s.semaphores()[lane], NUM_SOFTMAXXERS);
                } else if (lane < SEM_SCORES_ARRIVED + NUM_SOFTMAXXERS) {
                    kittens::init_semaphore(s.semaphores()[lane], 1);
                } else if (lane < SEM_NORM_ARRIVED + NUM_SOFTMAXXERS) {
                    kittens::init_semaphore(s.semaphores()[lane], 4 * Config::CLUSTER_SIZE);
                } else if (lane < SEM_NORM_Q_ARRIVED + 3 * NUM_SOFTMAXXERS) {
                    kittens::init_semaphore(s.semaphores()[lane], 4 * Config::CLUSTER_SIZE);
                } else {
                    kittens::init_semaphore(s.semaphores()[lane], 1);
                }
            }
            return SEM_COUNT;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const int q_batch   = instruction.indices[0];
            const int q_m_block = instruction.indices[1];
            const int q_head    = instruction.indices[2];
            const int k_batch   = instruction.indices[3];
            const int k_head    = instruction.indices[4];
            const int v_batch   = instruction.indices[5];
            const int v_head    = instruction.indices[6];
            const int cta_rank = kittens::cluster_ctarank();
            auto &q_gl = g.template gls<SRC_Q>();
            auto &k_gl = g.template gls<SRC_K>();
            auto &v_gl = g.template gls<SRC_V>();
            const int m_tile_base = q_m_block * NUM_SOFTMAXXERS * Config::CLUSTER_SIZE;
            int iters_per_task;
            if constexpr (CAUSAL) iters_per_task = m_tile_base + NUM_SOFTMAXXERS * Config::CLUSTER_SIZE;
            else                  iters_per_task = k_gl.depth() / Nb;

            if (kittens::warp::elect_leader()) {
                all_input_barrier_wait<Config>(g, instruction);

                #pragma unroll
                for (int i = 0; i < NUM_SOFTMAXXERS; i++)
                    s.page_wait(s.lid_to_pid(Q_LIDS[i]));
                #pragma unroll
                for (int i = 0; i < NUM_SOFTMAXXERS; i++) {
                    kittens::tma::cluster::load_async<kittens::dim::DEPTH, kittens::cache_policy::NORMAL>(
                        q_st(s, i), q_gl,
                        {q_batch, (m_tile_base + cta_rank * NUM_SOFTMAXXERS) + i, q_head, 0},
                        q_arrived(s, i), (uint16_t)(1 << cta_rank), 0);
                }

                #pragma unroll
                for (int i = 0; i < 2; i++)
                    s.page_wait(s.lid_to_pid(KV_LIDS[i]));
                int kv_idx = 0;
                int kv_phase = 1;
                for (int idx = 0; idx < iters_per_task; idx++) {
                    kittens::tma::cluster::wait(kv_finished(s, kv_idx), kv_phase);
                    kittens::tma::cluster::load_async<kittens::dim::DEPTH, kittens::cache_policy::NORMAL>(
                        kv_st(s, kv_idx), k_gl,
                        {k_batch, idx * Config::CLUSTER_SIZE + cta_rank, k_head, 0},
                        kv_arrived(s, kv_idx), (uint16_t)(1 << cta_rank), 0);
                    kv_idx++; if (kv_idx == LOAD_STAGES) { kv_idx = 0; kv_phase ^= 1; }

                    kittens::tma::cluster::wait(kv_finished(s, kv_idx), kv_phase);
                    kittens::tma::cluster::load_async<kittens::dim::DEPTH, kittens::cache_policy::NORMAL>(
                        reinterpret_cast<v_tile&>(kv_st(s, kv_idx)), v_gl,
                        {v_batch, idx, v_head, cta_rank},
                        kv_arrived(s, kv_idx), (uint16_t)(1 << cta_rank), 0);
                    kv_idx++; if (kv_idx == LOAD_STAGES) { kv_idx = 0; kv_phase ^= 1; }
                }
            } else if (kittens::warp::elect_leader_from_active()) {
                // Release unused pages
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
            const int cta_rank = kittens::cluster_ctarank();
            auto &k_gl = g.template gls<SRC_K>();
            int iters_per_task;
            if constexpr (CAUSAL) {
                const int q_m_block = s.instruction().indices[1];
                iters_per_task = q_m_block * NUM_SOFTMAXXERS * Config::CLUSTER_SIZE + NUM_SOFTMAXXERS * Config::CLUSTER_SIZE;
            } else {
                iters_per_task = k_gl.depth() / Nb;
            }

            if (cta_rank == 0 && kittens::warp::elect_leader()) {
                d_tt_scores tt_scores[NUM_SOFTMAXXERS];
                d_tt_outputs tt_outputs[NUM_SOFTMAXXERS];
                #pragma unroll
                for (int qid = 0; qid < NUM_SOFTMAXXERS; qid++) {
                    tt_scores[qid] = s.tensor_alloc.template allocate<d_tt_scores>(qid * (Nb + Db));
                    tt_outputs[qid] = s.tensor_alloc.template allocate<d_tt_outputs>(qid * (Nb + Db) + Nb);
                }
                s.tensor_wait();

                int kv_idx = 0;
                int kv_phase = 0;
                int norm_scores_phase = 0;

                #pragma unroll
                for (int qid = 0; qid < NUM_SOFTMAXXERS; qid++) {
                    kittens::tma::cluster::expect_bytes(q_arrived(s, qid), Config::CLUSTER_SIZE * sizeof(q_tile));
                    kittens::tma::cluster::wait(q_arrived(s, qid), 0);
                }

                // first QK
                int k_slot = kv_idx;
                kittens::tma::cluster::expect_bytes(kv_arrived(s, k_slot), Config::CLUSTER_SIZE * sizeof(k_tile));
                kittens::tma::cluster::wait(kv_arrived(s, k_slot), kv_phase);
                for (int qid = 0; qid < NUM_SOFTMAXXERS; qid++) {
                    kittens::mm2_ABt(tt_scores[qid], q_st(s, qid), kv_st(s, k_slot), kv_finished(s, k_slot));
                    kittens::detail::tcgen05::commit<Config::CLUSTER_SIZE>(scores_arrived(s, qid));
                }
                kv_idx++; if (kv_idx == LOAD_STAGES) { kv_idx = 0; kv_phase ^= 1; }

                // repeat PV then QK
                for (int idx = 0; idx < iters_per_task - 1; idx++) {
                    int v_slot = kv_idx;
                    kittens::tma::cluster::expect_bytes(kv_arrived(s, v_slot), Config::CLUSTER_SIZE * sizeof(v_tile));
                    kittens::tma::cluster::wait(kv_arrived(s, v_slot), kv_phase);
                    kv_idx++; if (kv_idx == LOAD_STAGES) { kv_idx = 0; kv_phase ^= 1; }
                    int k_slot = kv_idx;
                    auto *v_base = reinterpret_cast<const char*>(&kv_st(s, v_slot));
                    const v_quarter_tile *v_q[4];
                    #pragma unroll
                    for (int q = 0; q < 4; q++)
                        v_q[q] = reinterpret_cast<const v_quarter_tile*>(v_base + q * sizeof(v_quarter_tile));

                    #pragma unroll
                    for (int qid = 0; qid < NUM_SOFTMAXXERS; qid++) {
                        kittens::tma::cluster::wait(norm_scores_arrived(s, qid), norm_scores_phase);
                        if (idx == 0)
                            kittens::mm2_AB(tt_outputs[qid], d_tt_scores_bf_1q(tt_scores[qid].addr), *v_q[0]);
                        else
                            kittens::mma2_AB(tt_outputs[qid], d_tt_scores_bf_1q(tt_scores[qid].addr), *v_q[0]);
                        #pragma unroll
                        for (int q = 1; q < 4; q++) {
                            kittens::tma::cluster::wait(norm_scores_quarter_arrived(s, q-1, qid), norm_scores_phase);
                            if (q == 3)
                                kittens::mma2_AB(tt_outputs[qid], d_tt_scores_bf_1q(tt_scores[qid].addr + q * Nb/8), *v_q[q], kv_finished(s, v_slot));
                            else
                                kittens::mma2_AB(tt_outputs[qid], d_tt_scores_bf_1q(tt_scores[qid].addr + q * Nb/8), *v_q[q]);
                        }

                        if (qid == 0) {
                            kittens::tma::cluster::expect_bytes(kv_arrived(s, k_slot), Config::CLUSTER_SIZE * sizeof(k_tile));
                            kittens::tma::cluster::wait(kv_arrived(s, k_slot), kv_phase);
                        }
                        kittens::mm2_ABt(tt_scores[qid], q_st(s, qid), kv_st(s, k_slot), kv_finished(s, k_slot));
                        kittens::detail::tcgen05::commit<Config::CLUSTER_SIZE>(scores_arrived(s, qid));
                    }

                    kv_idx++; if (kv_idx == LOAD_STAGES) { kv_idx = 0; kv_phase ^= 1; }
                    norm_scores_phase ^= 1;
                }

                // last PV
                int v_slot = kv_idx;
                kittens::tma::cluster::expect_bytes(kv_arrived(s, v_slot), Config::CLUSTER_SIZE * sizeof(v_tile));
                kittens::tma::cluster::wait(kv_arrived(s, v_slot), kv_phase);

                auto *v_base = reinterpret_cast<const char*>(&kv_st(s, v_slot));
                const v_quarter_tile *v_q[4];
                #pragma unroll
                for (int q = 0; q < 4; q++)
                    v_q[q] = reinterpret_cast<const v_quarter_tile*>(v_base + q * sizeof(v_quarter_tile));

                #pragma unroll
                for (int qid = 0; qid < NUM_SOFTMAXXERS; qid++) {
                    kittens::tma::cluster::wait(norm_scores_arrived(s, qid), norm_scores_phase);
                    if (iters_per_task == 1)
                        kittens::mm2_AB(tt_outputs[qid], d_tt_scores_bf_1q(tt_scores[qid].addr), *v_q[0]);
                    else
                        kittens::mma2_AB(tt_outputs[qid], d_tt_scores_bf_1q(tt_scores[qid].addr), *v_q[0]);
                    #pragma unroll
                    for (int q = 1; q < 4; q++) {
                        kittens::tma::cluster::wait(norm_scores_quarter_arrived(s, q-1, qid), norm_scores_phase);
                        if (q == 3)
                            kittens::mma2_AB(tt_outputs[qid], d_tt_scores_bf_1q(tt_scores[qid].addr + q * Nb/8), *v_q[q], kv_finished(s, v_slot));
                        else
                            kittens::mma2_AB(tt_outputs[qid], d_tt_scores_bf_1q(tt_scores[qid].addr + q * Nb/8), *v_q[q]);
                    }
                    kittens::detail::tcgen05::commit<Config::CLUSTER_SIZE>(tile_arrived(s, qid));
                }
            }
        }
    };

    struct consumer {
        using consumer_group = kittens::group<kittens::WARPGROUP_WARPS * NUM_SOFTMAXXERS>;

        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const int q_m_block = instruction.indices[1];
            const int o_batch   = instruction.indices[7];
            const int o_m_block = instruction.indices[8];
            const int o_head    = instruction.indices[9];
            const int cta_rank = kittens::cluster_ctarank();
            auto &o_gl = g.template gls<DST_O>();
            auto &k_gl = g.template gls<SRC_K>();
            const int m_tile_base = q_m_block * NUM_SOFTMAXXERS * Config::CLUSTER_SIZE;
            const int o_m_tile_base = o_m_block * NUM_SOFTMAXXERS * Config::CLUSTER_SIZE;
            int iters_per_task;
            if constexpr (CAUSAL) iters_per_task = m_tile_base + NUM_SOFTMAXXERS * Config::CLUSTER_SIZE;
            else                  iters_per_task = k_gl.depth() / Nb;
            const int qid = kittens::warpgroup::groupid();

            static constexpr int CORR_TILE = 16;

            d_tt_scores tt_scores_handle = s.tensor_alloc.template allocate<d_tt_scores>(qid * (Nb + Db));
            d_tt_scores_bf tt_scores_bf_handle = d_tt_scores_bf(tt_scores_handle.addr);
            d_tt_outputs tt_output = s.tensor_alloc.template allocate<d_tt_outputs>(qid * (Nb + Db) + Nb);

            int scores_phase = 0;
            uint32_t score_tt_base = tt_scores_handle.addr + ((kittens::warpgroup::warpid() * 32) << 16);
            uint32_t score_bf_base = tt_scores_bf_handle.addr + ((kittens::warpgroup::warpid() * 32) << 16);
            uint32_t output_tt_base = tt_output.addr + ((kittens::warpgroup::warpid() * 32) << 16);
            float row_sum = 0.0f;
            float row_max = kittens::base_types::constants<float>::neg_infty();

            if constexpr (CAUSAL) {
            const int m_tile = m_tile_base + cta_rank * NUM_SOFTMAXXERS + qid;

            // Phase 1: unmasked tiles (idx < m_tile) — no causal masking needed
            for (int idx = 0; idx < m_tile; idx++) {
                float2 scores_reg[Nb / 2];
                kittens::wait(scores_arrived(s, qid), scores_phase);
                #pragma unroll
                for (int ii = 0; ii < Nb / 32; ii++) {
                    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                        : "=f"(scores_reg[ii*16+0].x), "=f"(scores_reg[ii*16+0].y), "=f"(scores_reg[ii*16+1].x), "=f"(scores_reg[ii*16+1].y), "=f"(scores_reg[ii*16+2].x), "=f"(scores_reg[ii*16+2].y), "=f"(scores_reg[ii*16+3].x), "=f"(scores_reg[ii*16+3].y),
                          "=f"(scores_reg[ii*16+4].x), "=f"(scores_reg[ii*16+4].y), "=f"(scores_reg[ii*16+5].x), "=f"(scores_reg[ii*16+5].y), "=f"(scores_reg[ii*16+6].x), "=f"(scores_reg[ii*16+6].y), "=f"(scores_reg[ii*16+7].x), "=f"(scores_reg[ii*16+7].y),
                          "=f"(scores_reg[ii*16+8].x), "=f"(scores_reg[ii*16+8].y), "=f"(scores_reg[ii*16+9].x), "=f"(scores_reg[ii*16+9].y), "=f"(scores_reg[ii*16+10].x), "=f"(scores_reg[ii*16+10].y), "=f"(scores_reg[ii*16+11].x), "=f"(scores_reg[ii*16+11].y),
                          "=f"(scores_reg[ii*16+12].x), "=f"(scores_reg[ii*16+12].y), "=f"(scores_reg[ii*16+13].x), "=f"(scores_reg[ii*16+13].y), "=f"(scores_reg[ii*16+14].x), "=f"(scores_reg[ii*16+14].y), "=f"(scores_reg[ii*16+15].x), "=f"(scores_reg[ii*16+15].y)
                        : "r"(score_tt_base + ii * 32));
                }

                const float SCALE_LOG2 = 1.44269504089f / sqrtf(float(Db));
                float row_max_old = row_max;

                float lm0 = kittens::base_types::constants<float>::neg_infty(), lm1 = kittens::base_types::constants<float>::neg_infty(), lm2 = kittens::base_types::constants<float>::neg_infty(), lm3 = kittens::base_types::constants<float>::neg_infty();
                float lm4 = kittens::base_types::constants<float>::neg_infty(), lm5 = kittens::base_types::constants<float>::neg_infty(), lm6 = kittens::base_types::constants<float>::neg_infty(), lm7 = kittens::base_types::constants<float>::neg_infty();
                #pragma unroll
                for (int j = 0; j < Nb / 2; j += 8) {
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(scores_reg[j+0].x), "f"(scores_reg[j+0].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm1) : "f"(lm1), "f"(scores_reg[j+1].x), "f"(scores_reg[j+1].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm2) : "f"(lm2), "f"(scores_reg[j+2].x), "f"(scores_reg[j+2].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm3) : "f"(lm3), "f"(scores_reg[j+3].x), "f"(scores_reg[j+3].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm4) : "f"(lm4), "f"(scores_reg[j+4].x), "f"(scores_reg[j+4].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm5) : "f"(lm5), "f"(scores_reg[j+5].x), "f"(scores_reg[j+5].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm6) : "f"(lm6), "f"(scores_reg[j+6].x), "f"(scores_reg[j+6].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm7) : "f"(lm7), "f"(scores_reg[j+7].x), "f"(scores_reg[j+7].y));
                }
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(lm1), "f"(lm2));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm4) : "f"(lm4), "f"(lm5), "f"(lm6));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(lm3), "f"(lm4));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(row_max) : "f"(row_max), "f"(lm0), "f"(lm7));

                float acc_scale = 1.0f;

                if (idx > 0) {
                    constexpr float rescale_threshold = 8.f;
                    float acc_scale_ = (row_max_old - row_max) * SCALE_LOG2;
                    if (acc_scale_ >= -rescale_threshold) {
                        row_max = row_max_old;
                        acc_scale = 1.0f;
                    } else {
                        acc_scale = exp2f(acc_scale_);
                    }

                    bool needs_rescale = __any_sync(0xFFFFFFFF, acc_scale < 1.0f);
                    if (needs_rescale) {
                        float2 corr_2 = {acc_scale, acc_scale};
                        #pragma unroll
                        for (int col = 0; col < Db; col += CORR_TILE) {
                            float2 o_reg[CORR_TILE / 2];
                            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                                : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                                  "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                                  "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                                  "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                                : "r"(output_tt_base + col));
                            kittens::tensor_load_wait();
                            #pragma unroll
                            for (int ii = 0; ii < CORR_TILE / 2; ii++)
                                o_reg[ii] = __fmul2_rn(o_reg[ii], corr_2);
                            asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
                                :: "f"(o_reg[0].x), "f"(o_reg[0].y), "f"(o_reg[1].x), "f"(o_reg[1].y),
                                   "f"(o_reg[2].x), "f"(o_reg[2].y), "f"(o_reg[3].x), "f"(o_reg[3].y),
                                   "f"(o_reg[4].x), "f"(o_reg[4].y), "f"(o_reg[5].x), "f"(o_reg[5].y),
                                   "f"(o_reg[6].x), "f"(o_reg[6].y), "f"(o_reg[7].x), "f"(o_reg[7].y),
                                   "r"(output_tt_base + col));
                        }
                        kittens::tensor_store_wait();
                    }
                }

                float neg_max_scaled = row_max * (-SCALE_LOG2);
                float2 neg_max_scaled_2 = {neg_max_scaled, neg_max_scaled};
                const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                constexpr int CONVERT_SIZE = 32;

                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    kittens::bf16_2 scores_bf_reg[CONVERT_SIZE / 2];
                    #pragma unroll
                    for (int jj = 0; jj < 16; jj++) {
                        int i = q * 16 + jj;
                        scores_reg[i] = __ffma2_rn(scores_reg[i], scale_2, neg_max_scaled_2);
                        scores_reg[i].x = exp2f(scores_reg[i].x);
                        scores_reg[i].y = exp2f(scores_reg[i].y);
                        scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[i]);
                    }
                    asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
                        :: "r"(*(uint32_t*)&scores_bf_reg[0]), "r"(*(uint32_t*)&scores_bf_reg[1]), "r"(*(uint32_t*)&scores_bf_reg[2]), "r"(*(uint32_t*)&scores_bf_reg[3]),
                           "r"(*(uint32_t*)&scores_bf_reg[4]), "r"(*(uint32_t*)&scores_bf_reg[5]), "r"(*(uint32_t*)&scores_bf_reg[6]), "r"(*(uint32_t*)&scores_bf_reg[7]),
                           "r"(*(uint32_t*)&scores_bf_reg[8]), "r"(*(uint32_t*)&scores_bf_reg[9]), "r"(*(uint32_t*)&scores_bf_reg[10]), "r"(*(uint32_t*)&scores_bf_reg[11]),
                           "r"(*(uint32_t*)&scores_bf_reg[12]), "r"(*(uint32_t*)&scores_bf_reg[13]), "r"(*(uint32_t*)&scores_bf_reg[14]), "r"(*(uint32_t*)&scores_bf_reg[15]),
                           "r"(score_bf_base + q * 16));
                    kittens::tensor_store_wait();
                    if (q == 0)
                        kittens::warp::tma::cluster::arrive(norm_scores_arrived(s, qid), 0);
                    else
                        kittens::warp::tma::cluster::arrive(norm_scores_quarter_arrived(s, q-1, qid), 0);
                }

                float2 ls0 = {0.0f, 0.0f}, ls1 = {0.0f, 0.0f};
                #pragma unroll
                for (int ii = 0; ii < Nb / 2; ii += 2) {
                    ls0 = __fadd2_rn(ls0, scores_reg[ii]);
                    ls1 = __fadd2_rn(ls1, scores_reg[ii + 1]);
                }
                ls0 = __fadd2_rn(ls0, ls1);
                row_sum = row_sum * acc_scale + ls0.x + ls0.y;

                scores_phase ^= 1;
            }

            // Phase 2: masked tiles (diagonal + fully masked) — causal masking applied
            for (int idx = m_tile; idx < iters_per_task; idx++) {
                float2 scores_reg[Nb / 2];
                kittens::wait(scores_arrived(s, qid), scores_phase);
                #pragma unroll
                for (int ii = 0; ii < Nb / 32; ii++) {
                    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                        : "=f"(scores_reg[ii*16+0].x), "=f"(scores_reg[ii*16+0].y), "=f"(scores_reg[ii*16+1].x), "=f"(scores_reg[ii*16+1].y), "=f"(scores_reg[ii*16+2].x), "=f"(scores_reg[ii*16+2].y), "=f"(scores_reg[ii*16+3].x), "=f"(scores_reg[ii*16+3].y),
                          "=f"(scores_reg[ii*16+4].x), "=f"(scores_reg[ii*16+4].y), "=f"(scores_reg[ii*16+5].x), "=f"(scores_reg[ii*16+5].y), "=f"(scores_reg[ii*16+6].x), "=f"(scores_reg[ii*16+6].y), "=f"(scores_reg[ii*16+7].x), "=f"(scores_reg[ii*16+7].y),
                          "=f"(scores_reg[ii*16+8].x), "=f"(scores_reg[ii*16+8].y), "=f"(scores_reg[ii*16+9].x), "=f"(scores_reg[ii*16+9].y), "=f"(scores_reg[ii*16+10].x), "=f"(scores_reg[ii*16+10].y), "=f"(scores_reg[ii*16+11].x), "=f"(scores_reg[ii*16+11].y),
                          "=f"(scores_reg[ii*16+12].x), "=f"(scores_reg[ii*16+12].y), "=f"(scores_reg[ii*16+13].x), "=f"(scores_reg[ii*16+13].y), "=f"(scores_reg[ii*16+14].x), "=f"(scores_reg[ii*16+14].y), "=f"(scores_reg[ii*16+15].x), "=f"(scores_reg[ii*16+15].y)
                        : "r"(score_tt_base + ii * 32));
                }

                // causal masking
                const float NEG_INFTY = kittens::base_types::constants<float>::neg_infty();
                int causal_col = (idx > m_tile) ? -1 : (int)kittens::warpgroup::laneid();
                #pragma unroll
                for (int k = 0; k < Nb / 2; k++) {
                    if (k * 2     > causal_col) scores_reg[k].x = NEG_INFTY;
                    if (k * 2 + 1 > causal_col) scores_reg[k].y = NEG_INFTY;
                }

                const float SCALE_LOG2 = 1.44269504089f / sqrtf(float(Db));
                float row_max_old = row_max;

                float lm0 = kittens::base_types::constants<float>::neg_infty(), lm1 = kittens::base_types::constants<float>::neg_infty(), lm2 = kittens::base_types::constants<float>::neg_infty(), lm3 = kittens::base_types::constants<float>::neg_infty();
                float lm4 = kittens::base_types::constants<float>::neg_infty(), lm5 = kittens::base_types::constants<float>::neg_infty(), lm6 = kittens::base_types::constants<float>::neg_infty(), lm7 = kittens::base_types::constants<float>::neg_infty();
                #pragma unroll
                for (int j = 0; j < Nb / 2; j += 8) {
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(scores_reg[j+0].x), "f"(scores_reg[j+0].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm1) : "f"(lm1), "f"(scores_reg[j+1].x), "f"(scores_reg[j+1].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm2) : "f"(lm2), "f"(scores_reg[j+2].x), "f"(scores_reg[j+2].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm3) : "f"(lm3), "f"(scores_reg[j+3].x), "f"(scores_reg[j+3].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm4) : "f"(lm4), "f"(scores_reg[j+4].x), "f"(scores_reg[j+4].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm5) : "f"(lm5), "f"(scores_reg[j+5].x), "f"(scores_reg[j+5].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm6) : "f"(lm6), "f"(scores_reg[j+6].x), "f"(scores_reg[j+6].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm7) : "f"(lm7), "f"(scores_reg[j+7].x), "f"(scores_reg[j+7].y));
                }
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(lm1), "f"(lm2));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm4) : "f"(lm4), "f"(lm5), "f"(lm6));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(lm3), "f"(lm4));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(row_max) : "f"(row_max), "f"(lm0), "f"(lm7));

                float acc_scale = 1.0f;
                {
                    constexpr float rescale_threshold = 8.f;
                    float acc_scale_ = (row_max_old - row_max) * SCALE_LOG2;
                    if (acc_scale_ >= -rescale_threshold) {
                        row_max = row_max_old;
                        acc_scale = 1.0f;
                    } else {
                        acc_scale = exp2f(acc_scale_);
                    }

                    bool needs_rescale = __any_sync(0xFFFFFFFF, acc_scale < 1.0f);
                    if (needs_rescale) {
                        float2 corr_2 = {acc_scale, acc_scale};
                        #pragma unroll
                        for (int col = 0; col < Db; col += CORR_TILE) {
                            float2 o_reg[CORR_TILE / 2];
                            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                                : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                                  "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                                  "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                                  "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                                : "r"(output_tt_base + col));
                            kittens::tensor_load_wait();
                            #pragma unroll
                            for (int ii = 0; ii < CORR_TILE / 2; ii++)
                                o_reg[ii] = __fmul2_rn(o_reg[ii], corr_2);
                            asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
                                :: "f"(o_reg[0].x), "f"(o_reg[0].y), "f"(o_reg[1].x), "f"(o_reg[1].y),
                                   "f"(o_reg[2].x), "f"(o_reg[2].y), "f"(o_reg[3].x), "f"(o_reg[3].y),
                                   "f"(o_reg[4].x), "f"(o_reg[4].y), "f"(o_reg[5].x), "f"(o_reg[5].y),
                                   "f"(o_reg[6].x), "f"(o_reg[6].y), "f"(o_reg[7].x), "f"(o_reg[7].y),
                                   "r"(output_tt_base + col));
                        }
                        kittens::tensor_store_wait();
                    }
                }

                float neg_max_scaled = row_max * (-SCALE_LOG2);
                float2 neg_max_scaled_2 = {neg_max_scaled, neg_max_scaled};
                const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                constexpr int CONVERT_SIZE = 32;

                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    kittens::bf16_2 scores_bf_reg[CONVERT_SIZE / 2];
                    #pragma unroll
                    for (int jj = 0; jj < 16; jj++) {
                        int i = q * 16 + jj;
                        scores_reg[i] = __ffma2_rn(scores_reg[i], scale_2, neg_max_scaled_2);
                        scores_reg[i].x = exp2f(scores_reg[i].x);
                        scores_reg[i].y = exp2f(scores_reg[i].y);
                        scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[i]);
                    }
                    asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
                        :: "r"(*(uint32_t*)&scores_bf_reg[0]), "r"(*(uint32_t*)&scores_bf_reg[1]), "r"(*(uint32_t*)&scores_bf_reg[2]), "r"(*(uint32_t*)&scores_bf_reg[3]),
                           "r"(*(uint32_t*)&scores_bf_reg[4]), "r"(*(uint32_t*)&scores_bf_reg[5]), "r"(*(uint32_t*)&scores_bf_reg[6]), "r"(*(uint32_t*)&scores_bf_reg[7]),
                           "r"(*(uint32_t*)&scores_bf_reg[8]), "r"(*(uint32_t*)&scores_bf_reg[9]), "r"(*(uint32_t*)&scores_bf_reg[10]), "r"(*(uint32_t*)&scores_bf_reg[11]),
                           "r"(*(uint32_t*)&scores_bf_reg[12]), "r"(*(uint32_t*)&scores_bf_reg[13]), "r"(*(uint32_t*)&scores_bf_reg[14]), "r"(*(uint32_t*)&scores_bf_reg[15]),
                           "r"(score_bf_base + q * 16));
                    kittens::tensor_store_wait();
                    if (q == 0)
                        kittens::warp::tma::cluster::arrive(norm_scores_arrived(s, qid), 0);
                    else
                        kittens::warp::tma::cluster::arrive(norm_scores_quarter_arrived(s, q-1, qid), 0);
                }

                float2 ls0 = {0.0f, 0.0f}, ls1 = {0.0f, 0.0f};
                #pragma unroll
                for (int ii = 0; ii < Nb / 2; ii += 2) {
                    ls0 = __fadd2_rn(ls0, scores_reg[ii]);
                    ls1 = __fadd2_rn(ls1, scores_reg[ii + 1]);
                }
                ls0 = __fadd2_rn(ls0, ls1);
                row_sum = row_sum * acc_scale + ls0.x + ls0.y;

                scores_phase ^= 1;
            }

            } else {

            for (int idx = 0; idx < iters_per_task; idx++) {
                float2 scores_reg[Nb / 2];
                kittens::wait(scores_arrived(s, qid), scores_phase);
                #pragma unroll
                for (int ii = 0; ii < Nb / 32; ii++) {
                    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                        : "=f"(scores_reg[ii*16+0].x), "=f"(scores_reg[ii*16+0].y), "=f"(scores_reg[ii*16+1].x), "=f"(scores_reg[ii*16+1].y), "=f"(scores_reg[ii*16+2].x), "=f"(scores_reg[ii*16+2].y), "=f"(scores_reg[ii*16+3].x), "=f"(scores_reg[ii*16+3].y),
                          "=f"(scores_reg[ii*16+4].x), "=f"(scores_reg[ii*16+4].y), "=f"(scores_reg[ii*16+5].x), "=f"(scores_reg[ii*16+5].y), "=f"(scores_reg[ii*16+6].x), "=f"(scores_reg[ii*16+6].y), "=f"(scores_reg[ii*16+7].x), "=f"(scores_reg[ii*16+7].y),
                          "=f"(scores_reg[ii*16+8].x), "=f"(scores_reg[ii*16+8].y), "=f"(scores_reg[ii*16+9].x), "=f"(scores_reg[ii*16+9].y), "=f"(scores_reg[ii*16+10].x), "=f"(scores_reg[ii*16+10].y), "=f"(scores_reg[ii*16+11].x), "=f"(scores_reg[ii*16+11].y),
                          "=f"(scores_reg[ii*16+12].x), "=f"(scores_reg[ii*16+12].y), "=f"(scores_reg[ii*16+13].x), "=f"(scores_reg[ii*16+13].y), "=f"(scores_reg[ii*16+14].x), "=f"(scores_reg[ii*16+14].y), "=f"(scores_reg[ii*16+15].x), "=f"(scores_reg[ii*16+15].y)
                        : "r"(score_tt_base + ii * 32));
                }

                const float SCALE_LOG2 = 1.44269504089f / sqrtf(float(Db));  // log2(e) / sqrt(Dqk)
                float row_max_old = row_max;

                // calculate row max
                float lm0 = kittens::base_types::constants<float>::neg_infty(), lm1 = kittens::base_types::constants<float>::neg_infty(), lm2 = kittens::base_types::constants<float>::neg_infty(), lm3 = kittens::base_types::constants<float>::neg_infty();
                float lm4 = kittens::base_types::constants<float>::neg_infty(), lm5 = kittens::base_types::constants<float>::neg_infty(), lm6 = kittens::base_types::constants<float>::neg_infty(), lm7 = kittens::base_types::constants<float>::neg_infty();
                #pragma unroll
                for (int j = 0; j < Nb / 2; j += 8) {
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(scores_reg[j+0].x), "f"(scores_reg[j+0].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm1) : "f"(lm1), "f"(scores_reg[j+1].x), "f"(scores_reg[j+1].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm2) : "f"(lm2), "f"(scores_reg[j+2].x), "f"(scores_reg[j+2].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm3) : "f"(lm3), "f"(scores_reg[j+3].x), "f"(scores_reg[j+3].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm4) : "f"(lm4), "f"(scores_reg[j+4].x), "f"(scores_reg[j+4].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm5) : "f"(lm5), "f"(scores_reg[j+5].x), "f"(scores_reg[j+5].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm6) : "f"(lm6), "f"(scores_reg[j+6].x), "f"(scores_reg[j+6].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm7) : "f"(lm7), "f"(scores_reg[j+7].x), "f"(scores_reg[j+7].y));
                }
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(lm1), "f"(lm2));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm4) : "f"(lm4), "f"(lm5), "f"(lm6));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(lm3), "f"(lm4));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(row_max) : "f"(row_max), "f"(lm0), "f"(lm7));

                float acc_scale = 1.0f;

                if (idx > 0) {
                    constexpr float rescale_threshold = 8.f;
                    float acc_scale_ = (row_max_old - row_max) * SCALE_LOG2;
                    if (acc_scale_ >= -rescale_threshold) {
                        row_max = row_max_old;
                        acc_scale = 1.0f;
                    } else {
                        acc_scale = exp2f(acc_scale_);
                    }

                    bool needs_rescale = __any_sync(0xFFFFFFFF, acc_scale < 1.0f);
                    if (needs_rescale) {
                        float2 corr_2 = {acc_scale, acc_scale};
                        #pragma unroll
                        for (int col = 0; col < Db; col += CORR_TILE) {
                            float2 o_reg[CORR_TILE / 2];
                            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                                : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                                  "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                                  "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                                  "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                                : "r"(output_tt_base + col));
                            kittens::tensor_load_wait();
                            #pragma unroll
                            for (int ii = 0; ii < CORR_TILE / 2; ii++)
                                o_reg[ii] = __fmul2_rn(o_reg[ii], corr_2);
                            asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
                                :: "f"(o_reg[0].x), "f"(o_reg[0].y), "f"(o_reg[1].x), "f"(o_reg[1].y),
                                   "f"(o_reg[2].x), "f"(o_reg[2].y), "f"(o_reg[3].x), "f"(o_reg[3].y),
                                   "f"(o_reg[4].x), "f"(o_reg[4].y), "f"(o_reg[5].x), "f"(o_reg[5].y),
                                   "f"(o_reg[6].x), "f"(o_reg[6].y), "f"(o_reg[7].x), "f"(o_reg[7].y),
                                   "r"(output_tt_base + col));
                        }
                        kittens::tensor_store_wait();
                    }
                }

                // scale, exp2, convert and store P in 4 quarters, signal after each
                float neg_max_scaled = row_max * (-SCALE_LOG2);
                float2 neg_max_scaled_2 = {neg_max_scaled, neg_max_scaled};
                const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                constexpr int CONVERT_SIZE = 32;

                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    kittens::bf16_2 scores_bf_reg[CONVERT_SIZE / 2];
                    #pragma unroll
                    for (int jj = 0; jj < 16; jj++) {
                        int i = q * 16 + jj;
                        scores_reg[i] = __ffma2_rn(scores_reg[i], scale_2, neg_max_scaled_2);
                        scores_reg[i].x = exp2f(scores_reg[i].x);
                        scores_reg[i].y = exp2f(scores_reg[i].y);
                        scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[i]);
                    }
                    asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
                        :: "r"(*(uint32_t*)&scores_bf_reg[0]), "r"(*(uint32_t*)&scores_bf_reg[1]), "r"(*(uint32_t*)&scores_bf_reg[2]), "r"(*(uint32_t*)&scores_bf_reg[3]),
                           "r"(*(uint32_t*)&scores_bf_reg[4]), "r"(*(uint32_t*)&scores_bf_reg[5]), "r"(*(uint32_t*)&scores_bf_reg[6]), "r"(*(uint32_t*)&scores_bf_reg[7]),
                           "r"(*(uint32_t*)&scores_bf_reg[8]), "r"(*(uint32_t*)&scores_bf_reg[9]), "r"(*(uint32_t*)&scores_bf_reg[10]), "r"(*(uint32_t*)&scores_bf_reg[11]),
                           "r"(*(uint32_t*)&scores_bf_reg[12]), "r"(*(uint32_t*)&scores_bf_reg[13]), "r"(*(uint32_t*)&scores_bf_reg[14]), "r"(*(uint32_t*)&scores_bf_reg[15]),
                           "r"(score_bf_base + q * 16));
                    kittens::tensor_store_wait();
                    if (q == 0)
                        kittens::warp::tma::cluster::arrive(norm_scores_arrived(s, qid), 0);
                    else
                        kittens::warp::tma::cluster::arrive(norm_scores_quarter_arrived(s, q-1, qid), 0);
                }

                // row sum
                float2 ls0 = {0.0f, 0.0f}, ls1 = {0.0f, 0.0f};
                #pragma unroll
                for (int ii = 0; ii < Nb / 2; ii += 2) {
                    ls0 = __fadd2_rn(ls0, scores_reg[ii]);
                    ls1 = __fadd2_rn(ls1, scores_reg[ii + 1]);
                }
                ls0 = __fadd2_rn(ls0, ls1);
                row_sum = row_sum * acc_scale + ls0.x + ls0.y;

                scores_phase ^= 1;
            }

            } // end if constexpr (CAUSAL)

            if (kittens::warpgroup::warpid() == 0 && kittens::warp::elect_leader())
                s.page_finish(s.lid_to_pid(Q_LIDS[qid]));

            // final normalization
            bool row_invalid = (row_sum == 0.0f) | (row_sum != row_sum);
            float inv_norm_s;
            asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_norm_s) : "f"(row_invalid ? 1.0f : row_sum));
            float2 inv_norm = {inv_norm_s, inv_norm_s};
            if (kittens::warpgroup::warpid() == 0 && kittens::warp::elect_leader())
                s.page_wait(s.lid_to_pid(O_LIDS[qid]));
            kittens::warpgroup::sync(qid + 1);
            kittens::wait(tile_arrived(s, qid), 0);

            constexpr int SUBTILE_COLS = 64;
            uint32_t base_addr = __cvta_generic_to_shared(&o_st(s, qid).data[0]);
            uint32_t row_offset = kittens::warpgroup::laneid() * SUBTILE_COLS * sizeof(kittens::bf16);

            for (int col = 0; col < Db; col += CORR_TILE) {
                float2 o_reg[CORR_TILE / 2];
                asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                    : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                      "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                      "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                      "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                    : "r"(output_tt_base + col));

                uint32_t row_base = base_addr + (col / SUBTILE_COLS) * (Mb * SUBTILE_COLS * sizeof(kittens::bf16))
                                  + row_offset + (col % SUBTILE_COLS) * sizeof(kittens::bf16);
                #pragma unroll
                for (int i = 0; i < CORR_TILE / 2; i++) {
                    kittens::bf16_2 tmp = __float22bfloat162_rn(__fmul2_rn(o_reg[i], inv_norm));
                    uint32_t addr = row_base + i * 4;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(addr ^ (((addr & 0x380) >> 7) << 4)), "r"(*(uint32_t*)&tmp));
                }
            }
            if (kittens::warpgroup::elect_leader()) all_reuse_barrier_wait<Config>(g, instruction);
            kittens::warpgroup::sync(qid + 1);
            kittens::warpgroup::tma::store_async<kittens::dim::DEPTH, kittens::cache_policy::EVICT_FIRST>(
                o_gl, o_st(s, qid),
                {o_batch, (o_m_tile_base + cta_rank * NUM_SOFTMAXXERS) + qid, o_head, 0});
            kittens::warpgroup::tma::store_async_wait();

            consumer_group::sync(4);
            if (consumer_group::elect_leader()) {
                #pragma unroll
                for (int i = 0; i < 2; i++)
                    s.page_finish(s.lid_to_pid(KV_LIDS[i]));
                #pragma unroll
                for (int i = 0; i < NUM_SOFTMAXXERS; i++)
                    s.page_finish(s.lid_to_pid(O_LIDS[i]));
                s.tensor_finish();
                all_barrier_arrive<Config>(g, instruction);
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {}
    };
};

} // namespace megakittens
