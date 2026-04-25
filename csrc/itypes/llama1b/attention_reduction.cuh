#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "utils.cuh"

namespace megakittens {

template <typename Config, typename Globals,
          int HEAD_DIM, int Q_HEADS_PER_INSTRUCTION, int MAX_PARTIALS,
          int SRC_LSE, int SRC_O_PARTIAL, int DST>
struct AttentionReduction {

    static constexpr int NUM_STAGES = 2;
    static_assert(NUM_STAGES <= 2);
    static constexpr int SHARED_DATA_PAGE = 0;

    using l_partial_sv = kittens::sv_fl<((MAX_PARTIALS + 15) / 16) * 16>;
    using o_sv = kittens::sv_fl<HEAD_DIM>;
    using o_rv = kittens::rv_fl<HEAD_DIM>;
    using o_final_sv = kittens::sv_bf<HEAD_DIM>;

    static constexpr size_t size_per_head = sizeof(l_partial_sv) + NUM_STAGES * sizeof(o_sv) + sizeof(o_final_sv);
    static constexpr size_t total_smem_needed = Q_HEADS_PER_INSTRUCTION * size_per_head;
    static_assert(total_smem_needed <= Config::PAGE_SIZE, "Required shared memory exceeds page size.");

    struct parsed_instruction {
        int layer_idx;
        int q_head_start_idx;
        int num_partials;
        int barrier_base;

        __device__ inline parsed_instruction(const instruction_t &instruction) {
            layer_idx        = instruction.indices[0];
            q_head_start_idx = instruction.indices[1];
            num_partials     = instruction.indices[2];
            barrier_base     = instruction.indices[3];
        }
        __device__ inline parsed_instruction(state_t<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    static constexpr int SEM_COUNT = Q_HEADS_PER_INSTRUCTION * (NUM_STAGES * 2 + 2);
    __device__ static constexpr int O_partial_sem_idx(int q_head, int stage, bool fin) { return q_head * NUM_STAGES * 2 + stage * 2 + fin; }
    __device__ static constexpr int L_partial_sem_idx(int q_head)                      { return Q_HEADS_PER_INSTRUCTION * NUM_STAGES * 2 + q_head; }
    __device__ static constexpr int Final_O_ready_sem_idx(int q_head)                  { return Q_HEADS_PER_INSTRUCTION * (NUM_STAGES * 2 + 1) + q_head; }

    __device__ static inline kittens::semaphore &O_partial_arrived(state_t<Config> &s, int q_head, int stage)  { return s.semaphores()[O_partial_sem_idx(q_head, stage, false)]; }
    __device__ static inline kittens::semaphore &O_partial_finished(state_t<Config> &s, int q_head, int stage) { return s.semaphores()[O_partial_sem_idx(q_head, stage, true)]; }
    __device__ static inline kittens::semaphore &L_partial_all_arrived(state_t<Config> &s, int q_head)  { return s.semaphores()[L_partial_sem_idx(q_head)]; }
    __device__ static inline kittens::semaphore &final_O_ready(state_t<Config> &s, int q_head) { return s.semaphores()[Final_O_ready_sem_idx(q_head)]; }

    __device__ static inline int data_pid(state_t<Config> &s) { return s.lid_to_pid(SHARED_DATA_PAGE); }

    __device__ static inline l_partial_sv &
    get_L_partial_smem(state_t<Config> &s, int q_head) {
        return s.pages[data_pid(s)].template as<l_partial_sv>(q_head * size_per_head);
    }
    __device__ static inline o_sv &
    get_O_partial_smem(state_t<Config> &s, int q_head, int stage) {
        int off = q_head * size_per_head + sizeof(l_partial_sv) + stage * sizeof(o_sv);
        return s.pages[data_pid(s)].template as<o_sv>(off);
    }
    __device__ static inline o_final_sv &
    get_O_final_smem(state_t<Config> &s, int q_head) {
        int off = q_head * size_per_head + sizeof(l_partial_sv) + NUM_STAGES * sizeof(o_sv);
        return s.pages[data_pid(s)].template as<o_final_sv>(off);
    }

    struct controller {
        __device__ __forceinline__ static int
        lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            if (query < Config::NUM_PAGES - 1)
                return query + 1;
            return 0;
        }
        __device__ __forceinline__ static int
        init_semaphores(const Globals &g, state_t<Config> &s) {
            int lid = kittens::laneid();
            if (lid < SEM_COUNT) {
                kittens::init_semaphore(s.semaphores()[lid], 1);
            }
            return SEM_COUNT;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            int laneid = kittens::laneid();
            if (laneid == 0) {
                s.page_wait(data_pid(s));
            } else if (laneid < Config::NUM_PAGES) {
                int pid = s.lid_to_pid(laneid);
                s.page_wait(pid);
                s.page_finish(pid);
            }
            kittens::warp::sync();

            if (kittens::warp::elect_leader()) {
                parsed_instruction inst{s};
                all_input_barrier_wait<Config>(g, s.instruction());

                for (int i = 0; i < Q_HEADS_PER_INSTRUCTION; i++) {
                    l_partial_sv &L_smem = get_L_partial_smem(s, i);
                    kittens::tma::expect(L_partial_all_arrived(s, i), L_smem);
                    kittens::tma::load_async<kittens::cache_policy::EVICT_FIRST>(
                        L_smem, g.template gls<SRC_LSE>(),
                        {inst.q_head_start_idx + i, 0},
                        L_partial_all_arrived(s, i));
                }

                for (int i = 0; i < inst.num_partials; i++) {
                    int stage = i % NUM_STAGES;
                    for (int j = 0; j < Q_HEADS_PER_INSTRUCTION; j++) {
                        o_sv &O_smem = get_O_partial_smem(s, j, stage);
                        if (i >= NUM_STAGES) {
                            int prev_phase = (i / NUM_STAGES - 1) % 2;
                            kittens::wait(O_partial_finished(s, j, stage), prev_phase);
                        }
                        kittens::tma::expect(O_partial_arrived(s, j, stage), O_smem);
                        kittens::tma::load_async<kittens::cache_policy::EVICT_FIRST>(
                            O_smem, g.template gls<SRC_O_PARTIAL>(),
                            {inst.q_head_start_idx + j, i, 0},
                            O_partial_arrived(s, j, stage));
                    }
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
            if (kittens::warpid() >= Q_HEADS_PER_INSTRUCTION) return;

            parsed_instruction inst{s};
            int q_head_local_idx = kittens::warpid();
            o_rv accumulated_out, current_out;
            float accumulated_lse = kittens::base_types::constants<float>::neg_infty(), current_lse;
            kittens::warp::zero(accumulated_out);
            kittens::wait(L_partial_all_arrived(s, q_head_local_idx), 0);
            l_partial_sv &L_smem = get_L_partial_smem(s, q_head_local_idx);
            for (int i = 0; i < inst.num_partials; i++) {
                int stage = i % NUM_STAGES;
                kittens::wait(O_partial_arrived(s, q_head_local_idx, stage),
                                    (i / NUM_STAGES) % 2);
                o_sv &O_smem = get_O_partial_smem(s, q_head_local_idx, stage);
                uint32_t src_ptr_L = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&L_smem.data[i]));
                kittens::move<float>::lds(current_lse, src_ptr_L);
                kittens::warp::load(current_out, O_smem);
                float max_lse = max(accumulated_lse, current_lse);
                float accumulated_exp = exp2f(accumulated_lse - max_lse);
                float current_exp = exp2f(current_lse - max_lse);
                float new_denom = accumulated_exp + current_exp;
                float accumulated_scale = accumulated_exp / new_denom;
                float current_scale = current_exp / new_denom;

                kittens::warp::mul(accumulated_out, accumulated_out, accumulated_scale);
                kittens::warp::mul(current_out, current_out, current_scale);
                kittens::warp::add(accumulated_out, accumulated_out, current_out);

                accumulated_lse = max_lse + log2f(new_denom);

                kittens::warp::arrive(O_partial_finished(s, q_head_local_idx, stage));
            }
            o_final_sv &O_final_smem = get_O_final_smem(s, q_head_local_idx);
            kittens::warp::store(O_final_smem, accumulated_out);
            kittens::warp::sync();
            kittens::warp::arrive(final_O_ready(s, q_head_local_idx));
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            parsed_instruction inst{s};
            if (kittens::laneid() < Q_HEADS_PER_INSTRUCTION) {
                int q_head_local_idx = kittens::laneid();
                o_final_sv &O_final_smem = get_O_final_smem(s, q_head_local_idx);
                kittens::wait(final_O_ready(s, q_head_local_idx), 0);
                kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                    g.template gls<DST>(), O_final_smem,
                    {inst.q_head_start_idx + q_head_local_idx});
                kittens::tma::store_async_wait();
            }
            kittens::warp::sync();
            if (kittens::warp::elect_leader()) {
                s.page_finish(data_pid(s));
                input_barrier_arrive<Config>(&g.barriers.raw_ptr[inst.barrier_base], 1);
            }
        }
    };
};

} // namespace megakittens
