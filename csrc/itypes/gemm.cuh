#pragma once

#include "kittens.cuh"

namespace megakittens {

template <typename Config, typename Globals, int SRC_A, int SRC_B, int DST_D>
struct Gemm {
    static constexpr int Mb = 256;
    static constexpr int Nb = 256;
    static constexpr int Kb = 64;
    static constexpr int NUM_CONSUMERS = 2;
    static constexpr int LOAD_PIPE_DEPTH = 4;
    static constexpr int EPI_PIPE_DEPTH = 8;
    static constexpr int NUM_D_TILES = 2;
    static constexpr int NUM_USED_PAGES = 6; // 4A2B (D reuses A0)

    using a_st_t = kittens::st_bf<Mb/2, Kb>;                  // 128×64
    using b_st_t = kittens::st_bf<Kb, Nb/2>;                  // 64×128
    using d_st_t = kittens::st_bf<Mb/2, Nb/EPI_PIPE_DEPTH>;   // 128×64
    using d_tt_t = kittens::tt<float, Mb/2, Nb>;              // 128×256 TMEM

    __device__ static inline kittens::semaphore &inputs_arrived  (state_t<Config> &s, int stage) { return s.semaphores()[stage]; }
    __device__ static inline kittens::semaphore &inputs_finished (state_t<Config> &s, int stage) { return s.semaphores()[LOAD_PIPE_DEPTH+stage]; }
    __device__ static inline kittens::semaphore &outputs_arrived (state_t<Config> &s)            { return s.semaphores()[2*LOAD_PIPE_DEPTH+0]; }

    __device__ static inline a_st_t &a_st(state_t<Config> &s, int stage, int cid) { return s.pages[s.lid_to_pid(stage)].template as<a_st_t>(cid*sizeof(a_st_t)); }
    __device__ static inline b_st_t &b_st(state_t<Config> &s, int stage)          { return s.pages[s.lid_to_pid(4+stage/2)].template as<b_st_t>((stage%2)*sizeof(b_st_t)); }
    __device__ static inline d_st_t &d_st(state_t<Config> &s, int cid, int slot)  { return s.pages[s.lid_to_pid(0)].template as<d_st_t>((cid*NUM_D_TILES+slot)*sizeof(d_st_t));}

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int lid) {
            return (lid + (Config::NUM_PAGES - NUM_USED_PAGES)) % Config::NUM_PAGES; // TODO really?
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
                #pragma unroll
                for (int i = NUM_USED_PAGES; i < Config::NUM_PAGES; i++) {
                    s.page_wait(s.lid_to_pid(i));
                    s.page_finish(s.lid_to_pid(i));
                }
                all_barrier_wait<Config>(g, instruction);

                for (int i = 0; i < num_iters + LOAD_PIPE_DEPTH; i++) {
                    const int stage = i % LOAD_PIPE_DEPTH;
                    if (i < LOAD_PIPE_DEPTH) {
                        s.page_wait(s.lid_to_pid(stage));
                        if (stage%2 == 0) s.page_wait(s.lid_to_pid(4+stage/2));
                    } else {
                        kittens::wait(inputs_finished(s, stage), ((i+LOAD_PIPE_DEPTH)/LOAD_PIPE_DEPTH)&0b1);
                    }
                    if (i < num_iters) {
                        #pragma unroll
                        for (int cid = 0; cid < NUM_CONSUMERS; cid++)
                            kittens::tma::cluster::load_async(a_st(s, stage, cid), a_gl, {(tile_row*2+cta_rank)*NUM_CONSUMERS+cid, i}, inputs_arrived(s, stage), (uint16_t)(1<<cta_rank), 0);
                        kittens::tma::cluster::load_async(b_st(s, stage), b_gl, {i, tile_col*2+cta_rank}, inputs_arrived(s, stage), (uint16_t)(1<<cta_rank), 0);
                    } else {
                        if (stage != 0) s.page_finish(s.lid_to_pid(stage));
                        if (stage%2 == 1) s.page_finish(s.lid_to_pid(4+(stage-1)/2));
                    }
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
            auto &d_gl = g.template gls<DST_D>();
            const int cid = kittens::warpgroup::groupid();

            d_tt_t d_tt = s.tensor_alloc.template allocate<d_tt_t>(cid * Nb);

            kittens::wait(outputs_arrived(s), 0);
            kittens::rt_bf<Mb/8, Nb/EPI_PIPE_DEPTH> d_reg[EPI_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < EPI_PIPE_DEPTH; i++)
                kittens::warpgroup::load_async(d_reg[i], d_tt.template subtile<kittens::tt<float, Mb/2, Nb/EPI_PIPE_DEPTH>>(0, (Nb/EPI_PIPE_DEPTH)*i));
            kittens::tensor_load_wait();
            consumer_group::sync(4);
            if (consumer_group::elect_leader()) s.tensor_finish();
        
            #pragma unroll
            for (int i = 0; i < EPI_PIPE_DEPTH; i++) {
                kittens::warpgroup::tma::store_async_read_wait<NUM_D_TILES - 1>();
                kittens::warpgroup::sync(cid + 1);
                kittens::warpgroup::store(d_st(s, cid, i%NUM_D_TILES), d_reg[i]);
                kittens::warpgroup::sync(cid + 1);
                kittens::warpgroup::tma::store_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(d_gl, d_st(s, cid, i%NUM_D_TILES), {(2*tile_row+cta_rank)*NUM_CONSUMERS+cid, EPI_PIPE_DEPTH*tile_col+i});
            }

            kittens::warpgroup::tma::store_async_wait(); // wait for all write to complete
            consumer_group::sync(4);
            if (consumer_group::elect_leader()) {
                s.page_finish(s.lid_to_pid(0));
                all_barrier_arrive<Config>(g, instruction);
            }
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) { } // unused
    };
};

} // namespace megakittens
