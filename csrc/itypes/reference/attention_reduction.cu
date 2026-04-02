#include "llama.cuh"
#include <limits>

using namespace kittens;
using namespace megakernel;

using globals = llama_1b_globals;

constexpr int Q_HEADS_PER_INSTRUCTION = 4;
constexpr int MAX_ATTN_PARTIALS = globals::sm_count;
constexpr int ROUNDED_MAX_ATTN_PARTIALS = ((MAX_ATTN_PARTIALS + 15) / 16) * 16;

using l_partial_sv = kittens::sv_fl<ROUNDED_MAX_ATTN_PARTIALS>;
using o_sv = kittens::sv_fl<globals::head_dim>;
using o_rv = kittens::rv_fl<globals::head_dim>;
using o_final_sv = kittens::sv_bf<globals::head_dim>;

template <typename Config, typename Globals> struct attention_reduction {
    static constexpr int opcode = OPCODE_AttentionReduction;
    static constexpr int prev_opcode = OPCODE_PartialAttention;
    static constexpr int NUM_STAGES = 2;
    static_assert(NUM_STAGES <= 2,
                  "Reduction NUM_STAGES must be less than or equal to 2");

    struct parsed_instruction {
        int layer_idx;
        int q_head_start_idx;
        int num_partials;
        int is_terminal;
        int reduction_list_length;
        int reduction_list[ROUNDED_MAX_ATTN_PARTIALS];

        __device__ inline parsed_instruction(megakernel::state<Config> &s) {
            layer_idx = s.instruction()[1];
            q_head_start_idx = s.instruction()[2];
            num_partials = s.instruction()[3];
            is_terminal = s.instruction()[4]; // not used
            reduction_list_length = s.instruction()[5];

#pragma unroll
            for (int k = 0; k < MAX_ATTN_PARTIALS; ++k) {
                if (k < reduction_list_length) {
                    reduction_list[k] = s.instruction()[6 + k];
                }
            }
        }
    };

    // --- kittens::semaphore Access Helpers ---
    __device__ static constexpr int
    O_partial_sem_idx(int q_head_local_idx, int stage, bool is_finished) {
        return q_head_local_idx * (NUM_STAGES * 2) + stage * 2 +
               (is_finished ? 1 : 0);
    }
    __device__ static constexpr int L_partial_sem_idx(int q_head_local_idx,
                                                      bool is_finished) {
        return (Q_HEADS_PER_INSTRUCTION * NUM_STAGES * 2) +
               q_head_local_idx * 2 + (is_finished ? 1 : 0);
    }
    __device__ static constexpr int
    Final_O_ready_sem_idx(int q_head_local_idx) {
        return (Q_HEADS_PER_INSTRUCTION * NUM_STAGES * 2) +
               (Q_HEADS_PER_INSTRUCTION * 2) + q_head_local_idx;
    }

    __device__ static inline kittens::semaphore &
    O_partial_arrived(megakernel::state<config> &s, int q_head_local_idx, int stage) {
        return s
            .semaphores()[O_partial_sem_idx(q_head_local_idx, stage, false)];
    }
    __device__ static inline kittens::semaphore &
    O_partial_finished(megakernel::state<config> &s, int q_head_local_idx, int stage) {
        return s.semaphores()[O_partial_sem_idx(q_head_local_idx, stage, true)];
    }
    __device__ static inline kittens::semaphore &
    L_partial_all_arrived(megakernel::state<config> &s, int q_head_local_idx) {
        return s.semaphores()[L_partial_sem_idx(q_head_local_idx, false)];
    }
    __device__ static inline kittens::semaphore &
    L_partial_all_finished(megakernel::state<config> &s, int q_head_local_idx) {
        return s.semaphores()[L_partial_sem_idx(q_head_local_idx, true)];
    }
    __device__ static inline kittens::semaphore &final_O_ready(megakernel::state<config> &s,
                                                      int q_head_local_idx) {
        return s.semaphores()[Final_O_ready_sem_idx(q_head_local_idx)];
    }

    // --- Shared Memory Page Management Helpers ---
    static constexpr int SHARED_DATA_PAGE =
        0; // Use only the first logical page

    __device__ static inline void wait_shared_page(megakernel::state<Config> &s) {
        if (kittens::warp::laneid() == 0) {
            s.wait_page_ready(s.pid(SHARED_DATA_PAGE));
        }
    }
    __device__ static inline void finish_shared_page(megakernel::state<Config> &s) {
        if (kittens::warp::laneid() == 0) {
            s.finish_page(s.pid(SHARED_DATA_PAGE), Config::NUM_CONSUMER_WARPS);
        }
    }

    // --- Shared Memory Layout and Access Helpers (Single Page) ---
    // Calculate the size needed for partials buffering
    static constexpr size_t size_per_head =
        sizeof(l_partial_sv) + NUM_STAGES * sizeof(o_sv) + sizeof(o_final_sv);
    static constexpr size_t total_smem_needed =
        Q_HEADS_PER_INSTRUCTION * size_per_head;
    static_assert(total_smem_needed <= config::PAGE_SIZE,
                  "Required shared memory exceeds configured page size.");

    __device__ static inline l_partial_sv &
    get_L_partial_smem(megakernel::state<config> &s, int q_head_local_idx) {
        int pid = s.pid(SHARED_DATA_PAGE);
        char *page_base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
        char *head_base_ptr = page_base_ptr + q_head_local_idx * size_per_head;
        return *reinterpret_cast<l_partial_sv *>(head_base_ptr);
    }
    __device__ static inline o_sv &
    get_O_partial_smem(megakernel::state<config> &s, int q_head_local_idx, int stage) {
        int pid = s.pid(SHARED_DATA_PAGE);
        char *page_base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
        char *head_base_ptr = page_base_ptr + q_head_local_idx * size_per_head;
        size_t offset = sizeof(l_partial_sv) + stage * sizeof(o_sv);
        return *reinterpret_cast<o_sv *>(head_base_ptr + offset);
    }
    __device__ static inline o_final_sv &
    get_O_final_smem(megakernel::state<config> &s, int q_head_local_idx) {
        int pid = s.pid(SHARED_DATA_PAGE);
        char *page_base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
        char *head_base_ptr = page_base_ptr + q_head_local_idx * size_per_head;
        size_t offset = sizeof(l_partial_sv) + NUM_STAGES * sizeof(o_sv);
        return *reinterpret_cast<o_final_sv *>(head_base_ptr + offset);
    }

    struct controller {
        static __device__ int
        release_lid(const Globals &g,
                    typename Config::instruction_t &instruction, int &query) {
            int ret_order[13] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0};
            return ret_order[query];
        }
        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
            for (int q_head = 0; q_head < Q_HEADS_PER_INSTRUCTION; ++q_head) {
                for (int stage = 0; stage < NUM_STAGES; stage++) {
                    init_semaphore(O_partial_arrived(s, q_head, stage), 0, 1);
                    init_semaphore(O_partial_finished(s, q_head, stage), 0, 1);
                }
                init_semaphore(L_partial_all_arrived(s, q_head), 0, 1);
                init_semaphore(L_partial_all_finished(s, q_head), 0, 1);

                init_semaphore(final_O_ready(s, q_head), 0, 1);
            }
            return 4 * ((NUM_STAGES * 2) + 3);
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            auto laneid = kittens::warp::laneid();

            if (laneid == 0) {
                wait_shared_page(s);
            } else if (laneid < Config::NUM_PAGES) {
                s.wait_page_ready(s.pid(laneid));
                s.finish_page(s.pid(laneid), Config::NUM_CONSUMER_WARPS);
            }
            kittens::warp::sync(); // Have to make sure lane 0 finished waiting
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            if (kittens::warp::laneid() == 0) {
#ifdef KITTENS_BLACKWELL
                s.wait_tensor_ready();
                arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
#endif

                parsed_instruction inst{s};

                s.record(megakernel::TEVENT_AT_GMEM_WAIT);
                while (*(volatile int *)&g.Bar[{inst.layer_idx, prev_opcode - 1,
                                                inst.q_head_start_idx + 0}] <
                           inst.num_partials ||
                       *(volatile int *)&g.Bar[{inst.layer_idx, prev_opcode - 1,
                                                inst.q_head_start_idx + 1}] <
                           inst.num_partials ||
                       *(volatile int *)&g.Bar[{inst.layer_idx, prev_opcode - 1,
                                                inst.q_head_start_idx + 2}] <
                           inst.num_partials ||
                       *(volatile int *)&g.Bar[{inst.layer_idx, prev_opcode - 1,
                                                inst.q_head_start_idx + 3}] <
                           inst.num_partials) {
                    __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                }
                s.record(megakernel::TEVENT_DONE_GMEM_WAIT);

                for (int i = 0; i < 4; ++i) {
                    l_partial_sv &L_smem = get_L_partial_smem(s, i);
                    kittens::tma::expect(L_partial_all_arrived(s, i), L_smem);
                    kittens::tma::load_async<cache_policy::EVICT_FIRST>(
                        L_smem, g.attn_lse_intermediates,
                        {0, 0, inst.q_head_start_idx + i, 0},
                        L_partial_all_arrived(s, i));
                }

                for (int i = 0; i < inst.num_partials; ++i) {
                    int stage = i % NUM_STAGES;
                    int cur_partial_idx = inst.reduction_list[i];
                    for (int j = 0; j < 4; ++j) {
                        o_sv &O_smem = get_O_partial_smem(s, j, stage);

                        if (i >= NUM_STAGES) {
                            int prev_phase = (i / NUM_STAGES - 1) % 2;
                            kittens::wait(O_partial_finished(s, j, stage), prev_phase);
                        }

                        kittens::tma::expect(O_partial_arrived(s, j, stage), O_smem);
                        kittens::tma::load_async<cache_policy::EVICT_FIRST>(
                            O_smem, g.attn_out_intermediates,
                            {0, inst.q_head_start_idx + j, cur_partial_idx, 0},
                            O_partial_arrived(s, j, stage));
                    }
                }
            }
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {

            if (kittens::warpid() < Q_HEADS_PER_INSTRUCTION) {

                parsed_instruction inst{s};
                int q_head_local_idx = kittens::warpid();

                o_rv accumulated_out;
                float accumulated_lse = -INFINITY;

                o_rv current_out;
                float current_lse;

                kittens::warp::zero(accumulated_out);

                kittens::warp::wait(L_partial_all_arrived(s, q_head_local_idx), 0);
                if (kittens::laneid() == 0)
                    s.record(megakernel::TEVENT_CONSUMER_START + 16 + kittens::warpid());
                l_partial_sv &L_smem = get_L_partial_smem(s, q_head_local_idx);

                // --- Reduction Pipeline ---
                for (int i = 0; i < inst.num_partials; ++i) {
                    int stage = i % NUM_STAGES;
                    kittens::warp::wait(O_partial_arrived(s, q_head_local_idx, stage),
                               (i / NUM_STAGES) % 2);

                    o_sv &O_smem =
                        get_O_partial_smem(s, q_head_local_idx, stage);

                    // Load cur L_partial value
                    int cur_partial_idx = inst.reduction_list[i];
                    uint32_t src_ptr_L =
                        static_cast<uint32_t>(__cvta_generic_to_shared(
                            &L_smem.data[cur_partial_idx]));
                    move<float>::lds(current_lse, src_ptr_L);
                    // Load O_partial_reg
                    kittens::warp::load(current_out, O_smem);

                    float max_lse = max(accumulated_lse, current_lse);

                    float accumulated_exp = exp2f(accumulated_lse - max_lse);
                    float current_exp = exp2f(current_lse - max_lse);

                    float new_denom = accumulated_exp + current_exp;

                    float accumulated_scale = accumulated_exp / new_denom;
                    float current_scale = current_exp / new_denom;

                    kittens::warp::mul(accumulated_out, accumulated_out,
                              accumulated_scale);
                    kittens::warp::mul(current_out, current_out, current_scale);
                    kittens::warp::add(accumulated_out, accumulated_out, current_out);

                    // Update LSE accumulator:
                    accumulated_lse = max_lse + log2f(new_denom);

                    kittens::warp::arrive(
                        O_partial_finished(s, q_head_local_idx, stage));
                }
                kittens::warp::arrive(L_partial_all_finished(s, q_head_local_idx));

                o_final_sv &O_final_smem =
                    get_O_final_smem(s, q_head_local_idx);
                kittens::warp::store(O_final_smem, accumulated_out);
                kittens::warp::sync();

                kittens::warp::arrive(final_O_ready(s, q_head_local_idx));
            }
        }
    };

    // Storer kittens::warp: Responsible for storing data from shared memory back to
    // global memory.
    struct storer {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            if (kittens::warp::laneid() < Q_HEADS_PER_INSTRUCTION) {
                int q_head_local_idx = kittens::warp::laneid();

                o_final_sv &O_final_smem =
                    get_O_final_smem(s, q_head_local_idx);
                kittens::wait(final_O_ready(s, q_head_local_idx), 0);
                if (kittens::warp::laneid() == 0) {
                    s.record(megakernel::TEVENT_OUTPUT_READY);
                }

                kittens::tma::store_async<cache_policy::NORMAL>(
                    g.attn_out, O_final_smem,
                    {0, 0, 0, inst.q_head_start_idx + q_head_local_idx});
                kittens::tma::store_async_wait();

                // atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1,
                // inst.q_head_start_idx + q_head_local_idx}], 1);
            }
            finish_shared_page(s);

            kittens::warp::sync();
            if (kittens::warp::laneid() == 0) {
                s.record(megakernel::TEVENT_AT_GMEM_STORE);
                // asm volatile("fence.acq_rel.gpu;");

                // simple signalling strat for now
                atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, 0}],
                          Q_HEADS_PER_INSTRUCTION);
                s.record(megakernel::TEVENT_DONE_GMEM_STORE);
            }
        }
    };
};