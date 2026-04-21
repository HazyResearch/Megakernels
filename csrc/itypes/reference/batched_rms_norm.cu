#include "llama.cuh"

using namespace kittens;
using namespace megakernel;

template <auto weights_ptr, auto outputs_ptr, int _opcode, typename gmem_waiter, typename Config, typename Globals>
struct rms_op {
    static constexpr int opcode = _opcode;
    static constexpr int REDUCTION_DIM_PER_WARP = Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

    static constexpr int SMEM_BYTES_PER_VEC = Globals::hidden_dim * sizeof(bf16);
    static constexpr int VECS_PER_PAGE = Config::PAGE_SIZE / SMEM_BYTES_PER_VEC;
    static_assert(VECS_PER_PAGE > 0, "Can't fit a vector in a page");

    // static constexpr int MAX_VECS = VECS_PER_PAGE * Config::NUM_PAGES;

    // static_assert(MAX_VECS * 2 + 1 <= Config::DYNAMIC_SEMAPHORES,
    //               "Not enough semaphores");

    static constexpr int WEIGHTS_PAGE = 0;  // 32kb shared page

    using weights_vec = sv_bf<Globals::hidden_dim>;

    struct parsed_instruction {
        int layer_idx;
        int num_items;
        int *local_batch_indices;
        __device__ inline parsed_instruction(typename Config::instruction_t &instruction) {
            layer_idx = instruction[1];
            num_items = instruction[2];
            local_batch_indices = instruction + 3;
        }
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}

        __device__ inline int num_used_pages() const {
            // ceil( (num_items + 1) / VECS_PER_PAGE)
            return ((num_items + 1) + VECS_PER_PAGE - 1) / VECS_PER_PAGE;
        }

        __device__ inline int global_batch_idx(const Globals &g, int i) const {
            return local_batch_idx_to_global_batch_idx(g, local_batch_indices[i]);
            // auto batch_size_per_device = g.batch_size / Globals::num_devices;
            // return batch_size_per_device * g.dev_idx +
            // local_batch_indices[i];
        }
    };

    // Semaphores
    __device__ static inline semaphore &weights_arrived(state<Config> &s) { return s.semaphores()[0]; }
    __device__ static inline semaphore &activations_arrived(state<Config> &s, int vec_idx) {
        return s.semaphores()[2 * vec_idx + 1];
    }
    __device__ static inline semaphore &outputs_arrived(state<Config> &s, int vec_idx) {
        return s.semaphores()[2 * vec_idx + 2];
    }

    __device__ static inline weights_vec &get_vec_ptr(state<Config> &s, int pid, int pos_in_page) {
        char *page_base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
        size_t offset = pos_in_page * SMEM_BYTES_PER_VEC;
        return *reinterpret_cast<weights_vec *>(page_base_ptr + offset);
    }

    __device__ static inline sv_bf<REDUCTION_DIM_PER_WARP> *get_vec_ptr_sliced_per_warp(state<Config> &s, int pid,
                                                                                        int pos_in_page) {
        auto &vec = get_vec_ptr(s, pid, pos_in_page);
        return reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP> *>(&vec);
    }

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query) {
            parsed_instruction inst{instruction};
            auto num_used_pages = inst.num_used_pages();
            auto num_pages_freed_immediately = Config::NUM_PAGES - num_used_pages;

            return (query + num_pages_freed_immediately) % Config::NUM_PAGES;
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            if(laneid() == 0) init_semaphore(weights_arrived(s), 1);

            if(laneid() < inst.num_items) {
                init_semaphore(activations_arrived(s, laneid()), 1);
                init_semaphore(outputs_arrived(s, laneid()), Config::NUM_CONSUMER_WARPS);
            }
            return 2 * inst.num_items + 1;
        }
    };
    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};

            warp::arrive(s.instruction_fetch_ready,
                         Config::NUM_CONSUMER_WARPS);  // these are fast, so we want to
                                                       // prefetch very early.

            if (laneid() == 0) {
                auto weight_pid = s.pid(WEIGHTS_PAGE);

                s.wait_page_ready(weight_pid);

                // RMS scale
                s.loader_record(LOAD2_EVENT);
                weights_vec &rms_scale = get_vec_ptr(s, weight_pid, 0);
                tma::expect(weights_arrived(s), rms_scale);
                auto &weights_global = g.*weights_ptr;
                tma::load_async(rms_scale, weights_global, {inst.layer_idx, 0}, weights_arrived(s));

                for (int i = 0; i < inst.num_items; i++) {
                    auto slot_with_rms_weight = i + 1;

                    auto page_idx = slot_with_rms_weight / VECS_PER_PAGE;
                    auto pos_in_page = slot_with_rms_weight % VECS_PER_PAGE;

                    auto pid = s.pid(page_idx);

                    auto local_batch_idx = inst.local_batch_indices[i];

                    if (pos_in_page == 0) {
                        s.wait_page_ready(pid);
                    }

                    if (i == 0) {
                        s.loader_record(WAIT_EVENT);
                    }

                    gmem_waiter::gmem_wait(g, s, inst, i);

                    if (i == 0) {
                        s.loader_record(READY_EVENT);
                    }

                    auto &ptr = get_vec_ptr(s, pid, pos_in_page);
                    auto &sem = activations_arrived(s, i);
                    s.loader_record(LOAD_EVENT);
                    tma::expect(sem, ptr);
                    tma::load_async(ptr, g.hidden_states[g.dev_idx], {local_batch_idx, 0}, sem);
                }

            } else if (laneid() >= inst.num_used_pages() && laneid() < Config::NUM_PAGES) {
                // Unused pages
                auto pid = s.pid(laneid());
                s.wait_page_ready(pid);

                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
            warp::sync();
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {
#ifdef KITTENS_BLACKWELL
            if (warp::laneid() == 0) {
                s.wait_tensor_ready();
                arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
            }
#endif
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            // Setup
            parsed_instruction inst{s};
            rv_fl<REDUCTION_DIM_PER_WARP> act_vec, copy_activations_vec,
                rms_scale_vec;  // 4096 / 16 = 256

            // multiply by rms scale
            wait(weights_arrived(s), 0);

            auto weights_smem = get_vec_ptr_sliced_per_warp(s, s.pid(WEIGHTS_PAGE), 0);
            warp::load(rms_scale_vec, weights_smem[warpid()]);
            warp::sync();

            for (int item_idx = 0; item_idx < inst.num_items; item_idx++) {
                auto index_with_rms_scale = item_idx + 1;
                auto page_idx = index_with_rms_scale / VECS_PER_PAGE;
                auto pos_in_page = index_with_rms_scale % VECS_PER_PAGE;

                auto pid = s.pid(page_idx);

                // Setup
                wait(activations_arrived(s, item_idx), 0);
                s.consumer_record(COMPUTE_EVENT);

                auto activations_smem = get_vec_ptr_sliced_per_warp(s, pid, pos_in_page);

                warp::load(act_vec, activations_smem[warpid()]);
                warp::sync();

                // Step 2: Apply RMS normalization
                warp::copy(copy_activations_vec, act_vec);  // cast to float
                warp::mul(copy_activations_vec, copy_activations_vec,
                          copy_activations_vec);  // square
                float partial_sum = warp::sum(copy_activations_vec);

                auto smem_rms_partial_sums = ((float *)s.scratch());
                // aggregate sums across the consumer warps
                if (laneid() == 0) {
                    smem_rms_partial_sums[warpid()] = partial_sum;
                }

                group<Config::NUM_CONSUMER_WARPS>::sync(0);

                float full_sum = 0;
                for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++) {
                    full_sum += smem_rms_partial_sums[i];
                }

                float variance = full_sum / (float)Globals::hidden_dim;
                float rms_scale = rsqrtf(variance + g.rms_norm_eps);

                warp::copy(copy_activations_vec, act_vec);  // unsquare
                warp::mul(copy_activations_vec, copy_activations_vec, rms_scale);
                warp::copy(act_vec, copy_activations_vec);

                warp::mul(act_vec, act_vec, rms_scale_vec);

                // Need to ensure storing here is correct!!!
                warp::store(activations_smem[warpid()], act_vec);
                s.consumer_record(STORE_EVENT);
                warp::sync();
                warp::arrive(outputs_arrived(s, item_idx));
            }
        }
    };

    struct storer {
        // Uses 4 full pages for outputs.
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};

            for (int i = 0; i < inst.num_items; i++) {
                auto index_with_rms_scale = i + 1;
                auto page_idx = index_with_rms_scale / VECS_PER_PAGE;
                auto pos_in_page = index_with_rms_scale % VECS_PER_PAGE;

                auto pid = s.pid(page_idx);

                auto local_batch_idx = inst.local_batch_indices[i];
                auto global_batch_idx = inst.global_batch_idx(g, i);

                wait(outputs_arrived(s, i), 0);
                s.storer_record(STORE_EVENT);
                auto &act_vec = get_vec_ptr(s, pid, pos_in_page);

                if (laneid() == 0) {
                    if constexpr (kittens::ducks::gl::all<std::remove_cvref_t<decltype(g.*outputs_ptr)>>)  // GL (LM
                                                                                                           // head RMS
                                                                                                           // norm)
                    {
                        tma::store_async<cache_policy::NORMAL>((g.*outputs_ptr), act_vec, {local_batch_idx, 0});
                    }

                    else  // PGL (attention and MLP RMS norm)
                    {
                        tma::store_async<cache_policy::NORMAL>((g.*outputs_ptr), act_vec, {global_batch_idx, 0},
                                                               g.dev_idx);
                    }

                    tma::store_async_read_wait();
                    s.storer_record(READY_EVENT);

                    if (pos_in_page == VECS_PER_PAGE - 1 || i == inst.num_items - 1) {
                        s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
                    }
                }

                tma::store_async_wait();
                warp::sync();
                gmem_waiter::inc_barrier(g, s, inst, local_batch_idx, global_batch_idx);
            }
        }
    };
};

struct attn_norm_gmem_waiter {
    template <typename Config, typename Globals, typename instruction_t>
    static __device__ inline void gmem_wait(const Globals &g, state<Config> &s, instruction_t &inst, int item_idx) {
        // int batch_block_idx = inst.global_batch_idx(g, item_idx) /
        //                       Globals::matmul_batch_block_size;

        int batch_block_idx = inst.local_batch_indices[item_idx] / Globals::matmul_batch_block_size;

        if (inst.layer_idx > 0) {
            int expected_val = Globals::num_output_blocks * g.num_devices;
            wait_on_barrier<Scope::SYS>(&g.Bar[g.dev_idx][{inst.layer_idx - 1, OPCODE_DownProjResidual - 1, batch_block_idx, 0}],
                expected_val, "attn_rms_norm",
                "dev=%d, layer=%d, batch_block=%d",
                g.dev_idx, inst.layer_idx - 1, batch_block_idx);
        }
    }

    template <typename Config, typename Globals, typename instruction_t>
    static __device__ inline void inc_barrier(const Globals &g, state<Config> &s, instruction_t &inst,
                                              int local_batch_idx, int global_batch_idx) {
        if (laneid() != 0) {
            return;
        }

        fence<Sem::RELEASE, Scope::SYS>();

        // Unfortunately, MC does not work with redAdd, so we
        // bump on all devices
        int batch_block_idx = global_batch_idx / Globals::matmul_batch_block_size;

        s.storer_record(STORE2_EVENT);
        for (int i = 0; i < g.num_devices; i++) {
            redAdd<Sem::RELAXED, Scope::SYS>(&g.Bar[i][{inst.layer_idx, OPCODE_AttnNorm - 1, batch_block_idx, 0}], 1);
        }
    }
};

template <typename Config, typename Globals>
struct attn_norm : rms_op<&Globals::attn_norm_weights, &Globals::rms_rope_intermediates, OPCODE_AttnNorm,
                          attn_norm_gmem_waiter, Config, Globals> {};

struct mlp_norm_gmem_waiter {
    template <typename Config, typename Globals, typename instruction_t>
    static __device__ inline void gmem_wait(const Globals &g, state<Config> &s, instruction_t &inst, int item_idx) {
        int batch_block_idx = inst.local_batch_indices[item_idx] / Globals::matmul_batch_block_size;
        int expected_val = Globals::num_output_blocks;
        wait_on_barrier<Scope::GPU>(&g.Bar[g.dev_idx][{inst.layer_idx, OPCODE_O_ProjResidual - 1, batch_block_idx, 0}],
            expected_val, "mlp_rms_norm",
            "dev=%d, layer=%d, batch_block=%d",
            g.dev_idx, inst.layer_idx, batch_block_idx);
    }

    template <typename Config, typename Globals, typename instruction_t>
    static __device__ inline void inc_barrier(const Globals &g, state<Config> &s, instruction_t &inst,
                                              int local_batch_idx, int global_batch_idx) {
        if (laneid() != 0) {
            return;
        }

        fence<Sem::RELEASE, Scope::SYS>();

        int batch_block_idx = global_batch_idx / Globals::matmul_batch_block_size;

        s.storer_record(STORE2_EVENT);
        for (int i = 0; i < g.num_devices; i++) {
            redAdd<Sem::RELAXED, Scope::SYS>(&g.Bar[i][{inst.layer_idx, OPCODE_MlpNorm - 1, batch_block_idx, 0}], 1);
        }
    }
};

template <typename Config, typename Globals>
struct mlp_norm : rms_op<&Globals::mlp_norm_weights, &Globals::rms_gate_intermediates, OPCODE_MlpNorm,
                         mlp_norm_gmem_waiter, Config, Globals> {};

struct lm_head_norm_gmem_waiter {
    template <typename Config, typename Globals, typename instruction_t>
    static __device__ inline void gmem_wait(const Globals &g, state<Config> &s, instruction_t &inst, int item_idx) {
        // int batch_block_idx = inst.global_batch_idx(g, item_idx) /
        //                       Globals::matmul_batch_block_size;

        int batch_block_idx = inst.local_batch_indices[item_idx] / Globals::matmul_batch_block_size;

        int expected_val = Globals::num_output_blocks * g.num_devices;
        wait_on_barrier<Scope::SYS>(&g.Bar[g.dev_idx][{Globals::num_hidden_layers - 1, OPCODE_DownProjResidual - 1, batch_block_idx, 0}],
            expected_val, "lm_head_rms_norm",
            "dev=%d, layer=%d, batch_block=%d",
            g.dev_idx, Globals::num_hidden_layers - 1, batch_block_idx);
    }

    template <typename Config, typename Globals, typename instruction_t>
    static __device__ inline void inc_barrier(const Globals &g, state<Config> &s, instruction_t &inst,
                                              int local_batch_idx, int global_batch_idx) {
        if (laneid() != 0) {
            return;
        }

        fence<Sem::RELEASE, Scope::GPU>();

        int batch_block_idx = local_batch_idx / Globals::matmul_batch_block_size;
        s.storer_record(STORE2_EVENT);
        redAdd<Sem::RELAXED, Scope::GPU>(&g.Bar[g.dev_idx][{0, OPCODE_LM_HeadNorm - 1, batch_block_idx, 0}], 1);
    }
};

template <typename Config, typename Globals>
struct lm_head_norm : rms_op<&Globals::lm_head_norm_weights, &Globals::rms_lm_head_intermediates, OPCODE_LM_HeadNorm,
                             lm_head_norm_gmem_waiter, Config, Globals> {};
