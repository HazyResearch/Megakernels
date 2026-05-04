#pragma once

#include "kittens.cuh"

namespace megakittens {

template <typename Config, typename Globals, int SRC_A, int SRC_B, int DST_D>
struct Gemmcta1 {
    static_assert(Config::CLUSTER_SIZE == 1, "Gemmcta1 requires CLUSTER_SIZE == 1");

    static constexpr int Mb = 256;
    static constexpr int Nb = 256;
    static constexpr int Kb = 64;
    static constexpr int NUM_CONSUMERS = 2;
    static constexpr int LOAD_PIPE_DEPTH = 3;
    static constexpr int EPI_PIPE_DEPTH = 8;
    static constexpr int NUM_D_TILES = 2;
    static constexpr int NUM_USED_PAGES = 6; // 3A3B (D reuses A0)

    using a_st_t = kittens::st_bf<Mb/2, Kb>;                  // 128×64
    using b_st_t = kittens::st_bf<Kb, Nb>;                    // 64×256 (full Nb, single CTA)
    using d_st_t = kittens::st_bf<Mb/2, Nb/EPI_PIPE_DEPTH>;   // 128×32
    using d_tt_t = kittens::tt<float, Mb/2, Nb>;              // 128×256 TMEM

    static constexpr int A_LIDS[LOAD_PIPE_DEPTH] = {0, 2, 4};
    static constexpr int B_LIDS[LOAD_PIPE_DEPTH] = {1, 3, 5};

    __device__ static inline kittens::semaphore &inputs_arrived  (state_t<Config> &s, int stage) { return s.semaphores()[stage]; }
    __device__ static inline kittens::semaphore &inputs_finished (state_t<Config> &s, int stage) { return s.semaphores()[LOAD_PIPE_DEPTH+stage]; }
    __device__ static inline kittens::semaphore &outputs_arrived (state_t<Config> &s)            { return s.semaphores()[2*LOAD_PIPE_DEPTH+0]; }

    __device__ static inline a_st_t &a_st(state_t<Config> &s, int stage, int cid) { return s.pages[s.lid_to_pid(A_LIDS[stage])].template as<a_st_t>(cid*sizeof(a_st_t)); }
    __device__ static inline b_st_t &b_st(state_t<Config> &s, int stage)          { return s.pages[s.lid_to_pid(B_LIDS[stage])].template as<b_st_t>(0); }
    __device__ static inline d_st_t &d_st(state_t<Config> &s, int cid, int slot)  { return s.pages[s.lid_to_pid(0)].template as<d_st_t>((cid*NUM_D_TILES+slot)*sizeof(d_st_t));}

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            static_assert(Config::NUM_PAGES == 7 && LOAD_PIPE_DEPTH == 3);
            const int num_iters = g.template gls<SRC_A>().cols() / Kb;
            switch (num_iters % LOAD_PIPE_DEPTH) {
                case 0: { constexpr int order[] = {6, 1, 2, 3, 4, 5, 0}; return order[query]; }
                case 1: { constexpr int order[] = {6, 2, 3, 4, 5, 1, 0}; return order[query]; }
                case 2: { constexpr int order[] = {6, 4, 5, 1, 2, 3, 0}; return order[query]; }
            }
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
            const int a_batch = instruction.indices[0];
            const int a_depth = instruction.indices[1];
            const int a_tile_m = instruction.indices[2];
            const int b_batch = instruction.indices[3];
            const int b_depth = instruction.indices[4];
            const int b_tile_n = instruction.indices[5];
            auto &a_gl = g.template gls<SRC_A>();
            auto &b_gl = g.template gls<SRC_B>();
            const int num_iters = a_gl.cols() / Kb;

            if (kittens::warp::elect_leader()) {
                all_input_barrier_wait<Config>(g, instruction);

                for (int i = 0; i < num_iters + LOAD_PIPE_DEPTH; i++) {
                    const int stage = i % LOAD_PIPE_DEPTH;
                    if (i < LOAD_PIPE_DEPTH) {
                        s.page_wait(s.lid_to_pid(A_LIDS[stage]));
                        s.page_wait(s.lid_to_pid(B_LIDS[stage]));
                    } else {
                        kittens::wait(inputs_finished(s, stage), ((i+LOAD_PIPE_DEPTH)/LOAD_PIPE_DEPTH)&0b1);
                    }
                    if (i < num_iters) {
                        #pragma unroll
                        for (int cid = 0; cid < NUM_CONSUMERS; cid++)
                            kittens::tma::load_async(a_st(s, stage, cid), a_gl, {a_batch, a_depth, a_tile_m*NUM_CONSUMERS+cid, i}, inputs_arrived(s, stage));
                        kittens::tma::load_async(b_st(s, stage), b_gl, {b_batch, b_depth, i, b_tile_n}, inputs_arrived(s, stage));
                    } else {
                        if (stage != 0) s.page_finish(s.lid_to_pid(A_LIDS[stage]));
                        s.page_finish(s.lid_to_pid(B_LIDS[stage]));
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
            auto &a_gl = g.template gls<SRC_A>();
            const int num_iters = a_gl.cols() / Kb;

            if (kittens::warp::elect_leader()) {
                d_tt_t d_tt[NUM_CONSUMERS];
                #pragma unroll
                for (int cid = 0; cid < NUM_CONSUMERS; cid++)
                    d_tt[cid] = s.tensor_alloc.template allocate<d_tt_t>(cid*Nb);
                s.tensor_wait();
                for (int i = 0; i < num_iters; i++) {
                    const int stage = i % LOAD_PIPE_DEPTH;
                    kittens::tma::expect_bytes(inputs_arrived(s, stage), NUM_CONSUMERS*sizeof(a_st_t) + sizeof(b_st_t));
                    kittens::wait(inputs_arrived(s, stage), (i/LOAD_PIPE_DEPTH)&0b1);
                    #pragma unroll
                    for (int cid = 0; cid < NUM_CONSUMERS; cid++) {
                        if (i == 0) kittens::mm_AB (d_tt[cid], a_st(s, stage, cid), b_st(s, stage), inputs_finished(s, stage));
                        else        kittens::mma_AB(d_tt[cid], a_st(s, stage, cid), b_st(s, stage), inputs_finished(s, stage));
                    }
                }
                kittens::tensor_commit<1>(outputs_arrived(s));
            }
        }
    };

    struct consumer {
        using consumer_group = kittens::group<kittens::WARPGROUP_WARPS * NUM_CONSUMERS>;

        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const int d_batch = instruction.indices[6],  d_depth = instruction.indices[7];
            const int d_tile_m = instruction.indices[8], d_tile_n = instruction.indices[9];
            auto &d_gl = g.template gls<DST_D>();
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
                kittens::warpgroup::tma::store_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(d_gl, d_st(s, cid, i%NUM_D_TILES), {d_batch, d_depth, d_tile_m*NUM_CONSUMERS+cid, EPI_PIPE_DEPTH*d_tile_n+i});
            }

            kittens::warpgroup::tma::store_async_wait(); // wait for all write to complete
            consumer_group::sync(4);
            if (consumer_group::elect_leader()) {
                s.page_finish(s.lid_to_pid(A_LIDS[0]));
                all_barrier_arrive<Config>(g, instruction);
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) { } // unused
    };
};

} // namespace megakittens
