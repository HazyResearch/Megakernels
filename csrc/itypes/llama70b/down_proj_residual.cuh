#pragma once

#include "kittens.cuh"

namespace megakittens {

// Fused matmul + reduce-scatter onto PGL output (faithful LLaMA-70B
// DownProjResidual, matching reference `matmul_adds.cu` with
// `reduce_scatter=true`).
//
// Each device holds its own K-shard of A and B. It computes the full (M, N)
// partial matmul tile-by-tile, then scatter-adds each 128-row sub-tile to
// the owning peer device's slice of the PGL output D via
// `tma::store_add_async(d_pgl.gls[target_dev], ...)`. After all N kernels
// complete, each device holds only its own M-shard of
// sum_k A_k @ B_k — i.e. the K-dim all-reduce of the matmul is fused with
// the M-dim scatter.
//
// target_dev is computed from the global output row index:
//   global_row_128 = (2*tile_row + cta_rank) * NUM_CONSUMERS + cid
//   target_dev     = global_row_128 / rows_per_dev_128
//   local_row_128  = global_row_128 % rows_per_dev_128
// where rows_per_dev_128 = d_pgl.gls[0].rows() / (Mb/2) (each peer's
// scattered shape is M_total / NUM_DEVICES rows).
//
// "Residual" semantics are achieved by pre-initializing D to the residual
// value; the store_add_async accumulates the matmul on top.
template <typename Config, typename Globals, int SRC_A, int SRC_B, int DST_D>
struct DownProjResidual {
    static constexpr int Mb = 256;
    static constexpr int Nb = 256;
    static constexpr int Kb = 64;
    static constexpr int NUM_CONSUMERS = 2;
    static constexpr int LOAD_PIPE_DEPTH = 4;
    static constexpr int EPI_PIPE_DEPTH = 8;
    static constexpr int NUM_D_TILES = 2;
    static constexpr int NUM_USED_PAGES = 6; // 4A 2B (D reuses A0) — no C scratch needed

    using a_st_t = kittens::st_bf<Mb/2, Kb>;                  // 128x64
    using b_st_t = kittens::st_bf<Kb, Nb/2>;                  // 64x128
    using d_st_t = kittens::st_bf<Mb/2, Nb/EPI_PIPE_DEPTH>;   // 128x32
    using d_tt_t = kittens::tt<float, Mb/2, Nb>;              // 128x256 TMEM

    static constexpr int A_LIDS[LOAD_PIPE_DEPTH] = {0, 2, 3, 5};
    static constexpr int B_LIDS[2] = {1, 4};

    __device__ static inline kittens::semaphore &inputs_arrived  (state_t<Config> &s, int stage) { return s.semaphores()[stage]; }
    __device__ static inline kittens::semaphore &inputs_finished (state_t<Config> &s, int stage) { return s.semaphores()[LOAD_PIPE_DEPTH+stage]; }
    __device__ static inline kittens::semaphore &outputs_arrived (state_t<Config> &s)            { return s.semaphores()[2*LOAD_PIPE_DEPTH+0]; }

    __device__ static inline a_st_t &a_st(state_t<Config> &s, int stage, int cid) { return s.pages[s.lid_to_pid(A_LIDS[stage])].template as<a_st_t>(cid*sizeof(a_st_t)); }
    __device__ static inline b_st_t &b_st(state_t<Config> &s, int stage)          { return s.pages[s.lid_to_pid(B_LIDS[stage/2])].template as<b_st_t>((stage%2)*sizeof(b_st_t)); }
    __device__ static inline d_st_t &d_st(state_t<Config> &s, int cid, int slot)  { return s.pages[s.lid_to_pid(0)].template as<d_st_t>((cid*NUM_D_TILES+slot)*sizeof(d_st_t));}

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            static_assert(Config::NUM_PAGES == 7 && LOAD_PIPE_DEPTH == 4);
            const int num_iters = g.template gls<SRC_A>().cols() / Kb;
            switch (num_iters % LOAD_PIPE_DEPTH) {
                case 0: case 1: { constexpr int order[] = {6, 2, 1, 3, 5, 4, 0}; return order[query]; }
                case 2:         { constexpr int order[] = {6, 3, 5, 4, 2, 1, 0}; return order[query]; }
                case 3:         { constexpr int order[] = {6, 5, 4, 2, 1, 3, 0}; return order[query]; }
            }
            return 0;
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            const int lane_id = kittens::laneid();
            if (lane_id < LOAD_PIPE_DEPTH) {
                kittens::init_semaphore(inputs_arrived(s, lane_id), 1);
                kittens::init_semaphore(inputs_finished(s, lane_id), NUM_CONSUMERS);
            } else if (lane_id == LOAD_PIPE_DEPTH) {
                kittens::init_semaphore(outputs_arrived(s), 1);
            }
            return 2*LOAD_PIPE_DEPTH + 1;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const int tile_row = instruction.indices[0];
            const int tile_col = instruction.indices[1];
            const int cta_rank = kittens::cluster_ctarank();
            auto &a_gl = g.template gls<SRC_A>();
            auto &b_gl = g.template gls<SRC_B>();
            const int num_iters = a_gl.cols() / Kb;

            if (kittens::warp::elect_leader()) {
                all_input_barrier_wait<Config>(g, instruction);

                for (int i = 0; i < num_iters + LOAD_PIPE_DEPTH; i++) {
                    const int stage = i % LOAD_PIPE_DEPTH;
                    if (i < LOAD_PIPE_DEPTH) {
                        s.page_wait(s.lid_to_pid(A_LIDS[stage]));
                        if (stage%2 == 0) s.page_wait(s.lid_to_pid(B_LIDS[stage/2]));
                    } else {
                        kittens::wait(inputs_finished(s, stage), ((i+LOAD_PIPE_DEPTH)/LOAD_PIPE_DEPTH)&0b1);
                    }
                    if (i < num_iters) {
                        #pragma unroll
                        for (int cid = 0; cid < NUM_CONSUMERS; cid++)
                            kittens::tma::cluster::load_async(a_st(s, stage, cid), a_gl, {(tile_row*2+cta_rank)*NUM_CONSUMERS+cid, i}, inputs_arrived(s, stage), (uint16_t)(1<<cta_rank), 0);
                        kittens::tma::cluster::load_async(b_st(s, stage), b_gl, {i, tile_col*2+cta_rank}, inputs_arrived(s, stage), (uint16_t)(1<<cta_rank), 0);
                    } else {
                        if (stage != 0) s.page_finish(s.lid_to_pid(A_LIDS[stage]));
                        if (stage%2 == 1) s.page_finish(s.lid_to_pid(B_LIDS[(stage-1)/2]));
                    }
                }
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
            const int cta_rank = kittens::cluster_ctarank();
            auto &a_gl = g.template gls<SRC_A>();
            const int num_iters = a_gl.cols() / Kb;

            if (cta_rank == 0 && kittens::warp::elect_leader()) {
                d_tt_t d_tt[NUM_CONSUMERS];
                #pragma unroll
                for (int cid = 0; cid < NUM_CONSUMERS; cid++)
                    d_tt[cid] = s.tensor_alloc.template allocate<d_tt_t>(cid*Nb);
                s.tensor_wait();
                for (int i = 0; i < num_iters; i++) {
                    const int stage = i % LOAD_PIPE_DEPTH;
                    kittens::tma::expect_bytes(inputs_arrived(s, stage), 2*NUM_CONSUMERS*sizeof(a_st_t) + 2*sizeof(b_st_t));
                    kittens::wait(inputs_arrived(s, stage), (i/LOAD_PIPE_DEPTH)&0b1);
                    #pragma unroll
                    for (int cid = 0; cid < NUM_CONSUMERS; cid++) {
                        if (i == 0) kittens::mm2_AB (d_tt[cid], a_st(s, stage, cid), b_st(s, stage), inputs_finished(s, stage));
                        else        kittens::mma2_AB(d_tt[cid], a_st(s, stage, cid), b_st(s, stage), inputs_finished(s, stage));
                    }
                }
                kittens::tensor_commit<2>(outputs_arrived(s));
            }
        }
    };

    struct consumer {
        using consumer_group = kittens::group<kittens::WARPGROUP_WARPS * NUM_CONSUMERS>;

        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const int tile_row = instruction.indices[0], tile_col = instruction.indices[1];
            const int cta_rank = kittens::cluster_ctarank();
            auto &d_pgl = g.template pgls<DST_D>();
            const int cid = kittens::warpgroup::groupid();

            d_tt_t d_tt = s.tensor_alloc.template allocate<d_tt_t>(cid * Nb);

            kittens::wait(outputs_arrived(s), 0);
            kittens::rt_bf<Mb/8, Nb/EPI_PIPE_DEPTH> d_reg[EPI_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < EPI_PIPE_DEPTH; i++)
                kittens::warpgroup::load_async(d_reg[i], d_tt.template subtile<kittens::tt<float, Mb/2, Nb/EPI_PIPE_DEPTH>>(0, (Nb/EPI_PIPE_DEPTH)*i));
            kittens::tensor_load_wait();
            if (consumer_group::elect_leader()) all_reuse_barrier_wait<Config>(g, instruction);
            consumer_group::sync(4);
            if (consumer_group::elect_leader()) s.tensor_finish();

            #pragma unroll
            for (int i = 0; i < EPI_PIPE_DEPTH; i++) {
                kittens::warpgroup::tma::store_async_read_wait<NUM_D_TILES - 1>();
                kittens::warpgroup::sync(cid + 1);
                kittens::warpgroup::store(d_st(s, cid, i%NUM_D_TILES), d_reg[i]);
                kittens::warpgroup::sync(cid + 1);
                // Reduce-scatter: route each 128-row sub-tile to the device
                // that owns that M-slice. rows_per_dev_128 is d_pgl.gls[0].rows() / 128
                // because the scattered layout gives each peer M_total/NUM_DEVICES rows.
                if (kittens::warpgroup::elect_leader()) {
                    const int global_row_128 = (2*tile_row + cta_rank) * NUM_CONSUMERS + cid;
                    const int rows_per_dev_128 = d_pgl.gls[0].rows() / (Mb/2);
                    const int target_dev = global_row_128 / rows_per_dev_128;
                    const int local_row_128 = global_row_128 % rows_per_dev_128;
                    kittens::tma::store_add_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                        d_pgl.gls[target_dev], d_st(s, cid, i%NUM_D_TILES),
                        {local_row_128, EPI_PIPE_DEPTH*tile_col+i});
                }
            }

            kittens::warpgroup::tma::store_async_wait();
            consumer_group::sync(4);
            if (consumer_group::elect_leader()) {
                s.page_finish(s.lid_to_pid(A_LIDS[0])); // D's staging page (reused)
                all_barrier_arrive<Config>(g, instruction);
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) { } // unused
    };
};

} // namespace megakittens
