#pragma once

#include "kittens.cuh"

namespace megakittens {

constexpr float NEG_INFTY = std::bit_cast<float>(0xFF800000);

template <typename Config, typename Globals, int SRC_Q, int SRC_K, int SRC_V, int DST_O>
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
            if (kittens::laneid() < Config::NUM_PAGES) {
                int pid = s.lid_to_pid(kittens::laneid());
                s.page_wait(pid);
                s.page_finish(pid);
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
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {}
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {}
    };
};

} // namespace megakittens
