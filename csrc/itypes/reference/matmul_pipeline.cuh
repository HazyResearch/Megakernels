#pragma once

#include "llama.cuh"

static constexpr int PIPELINE_K_DIM = 64;

template <typename Config, typename Globals, typename parsed_instruction, typename gmem_waiter, auto A_Ptr, auto B_Ptr,
          int _num_iters>
struct matmul_pipeline {
    static_assert(Config::PAGE_SIZE == 32768);
    static_assert(Config::SCRATCH_BYTES >= 8192);

    static constexpr int INPUT_PIPELINE_STAGES = 4;
    static constexpr int MATMULS_PER_STAGE = 2;

    static_assert(_num_iters % INPUT_PIPELINE_STAGES == 0, "Invalid number of iters");

#ifdef KITTENS_BLACKWELL
    using a_st = st_bf<128, PIPELINE_K_DIM>;
#else
    using a_st = st_bf<64, PIPELINE_K_DIM>;
#endif
    using b_st = st_bf<256, PIPELINE_K_DIM>;

#ifdef KITTENS_BLACKWELL
    static constexpr int SEM_COUNT = 2 * INPUT_PIPELINE_STAGES + 1;
#else
    static constexpr int SEM_COUNT = 2 * INPUT_PIPELINE_STAGES;
#endif

    __device__ static inline int get_a_page(state<Config> &s, int stage) { return s.pid(3 * (stage / 2)); }
    __device__ static inline int get_b_page(state<Config> &s, int stage) {
        return s.pid(1 + 3 * (stage / 2) + (stage & 1));
    }
    __device__ static inline a_st (&get_a_tile(state<Config> &s, int stage))[2] {
        return reinterpret_cast<a_st(*)[2]>(&s.pages[get_a_page(s, stage)])[stage & 1];
    }
    __device__ static inline b_st &get_b_tile(state<Config> &s, int stage) {
        return reinterpret_cast<b_st &>(s.pages[get_b_page(s, stage)]);
    }

    __device__ static inline int get_output_lid(int wg) {
        constexpr int reverse_release_order[6] = {1, 0, 2, 4, 3, 5};
        return reverse_release_order[wg];
    }

    // pid
    __device__ static inline int get_output_page(state<Config> &s, int wg) { return s.pid(get_output_lid(wg)); }
    __device__ static inline st_bf<64, 256> &get_output_tile(state<Config> &s, int wg) {
        return reinterpret_cast<st_bf<64, 256> &>(s.pages[get_output_page(s, wg)]);
    }

    __device__ static inline semaphore &inputs_arrived(state<Config> &s, int stage) { return s.semaphores()[stage]; }
    __device__ static inline semaphore &inputs_finished(state<Config> &s, int stage) {
        return s.semaphores()[INPUT_PIPELINE_STAGES + stage];
    }
#ifdef KITTENS_BLACKWELL
    __device__ static inline semaphore &outputs_arrived(state<Config> &s) {
        return s.semaphores()[2 * INPUT_PIPELINE_STAGES];
    }
#endif

    __device__ static inline void loader_input_wait(state<Config> &s, int iter) {
        int input_stage = iter % INPUT_PIPELINE_STAGES;
        wait(inputs_finished(s, input_stage), (iter % (2 * INPUT_PIPELINE_STAGES)) < INPUT_PIPELINE_STAGES);
    }

    template <int _pages_in_reserve = 0>
    __device__ static inline int release_lid(const Globals &g, typename Config::instruction_t &instruction,
                                             int &query) {
        static_assert(INPUT_PIPELINE_STAGES == 4, "INPUT_PIPELINE_STAGES must be 3");

        parsed_instruction inst{instruction};

        constexpr int remainder = _num_iters % INPUT_PIPELINE_STAGES;

        // this pipeline only works with a number of iters that is a multiple of
        // 4
        if constexpr (remainder == 0) {
            if constexpr (_pages_in_reserve % 6 == 0) {
                constexpr int ret_order[6] = {1, 0, 2, 4, 3, 5};
                return ret_order[query];
            } else if constexpr (_pages_in_reserve % 6 == 1) {
                constexpr int ret_order[6] = {0, 2, 4, 3, 5, 1};
                return ret_order[query];
            } else if constexpr (_pages_in_reserve % 6 == 2) {
                constexpr int ret_order[6] = {2, 4, 3, 5, 1, 0};
                return ret_order[query];
            } else if constexpr (_pages_in_reserve % 6 == 3) {
                constexpr int ret_order[6] = {4, 3, 5, 1, 0, 2};
                return ret_order[query];
            } else if constexpr (_pages_in_reserve % 6 == 4) {
                constexpr int ret_order[6] = {3, 5, 1, 0, 2, 4};
                return ret_order[query];
            } else if constexpr (_pages_in_reserve % 6 == 5) {
                constexpr int ret_order[6] = {5, 1, 0, 2, 4, 3};
                return ret_order[query];
            } else {
                static_assert(remainder == -1, "Invalid number of iters");
            }
        } else {
            static_assert(remainder == -1, "Invalid number of iters");
        }
    }

    __device__ static inline int init_semaphores(state<Config> &s) {
        if(laneid() < INPUT_PIPELINE_STAGES) {
            init_semaphore(inputs_arrived(s, laneid()), 1);
            init_semaphore(inputs_finished(s, laneid()), MATMULS_PER_STAGE);
        }
#ifdef KITTENS_BLACKWELL
        if(laneid() == 0) init_semaphore(outputs_arrived(s), 1);
#endif
        return SEM_COUNT;
    }

    template <bool _mark_ready_for_instruction = false, int _axis = dim::ROW>
    __device__ static inline void loader_loop(state<Config> &s, const Globals &g, int layer_idx = 0) {
        parsed_instruction inst{s};

        if (laneid() == 0) {
            for (int iter = 0; iter < _num_iters; iter++) {
                int input_stage = iter % INPUT_PIPELINE_STAGES;
                if constexpr (_mark_ready_for_instruction) {
                    if (iter == _num_iters - 3) {
                        arrive(s.instruction_fetch_ready, Config::NUM_CONSUMER_WARPS);
                    }
                }
                loader_input_wait(s, iter);

                auto &sem = inputs_arrived(s, input_stage);
                tma::expect_bytes(sem, sizeof(bf16) * ((a_st::rows * MATMULS_PER_STAGE) + b_st::rows) * PIPELINE_K_DIM);

                int a_page = get_a_page(s, input_stage), b_page = get_b_page(s, input_stage);
                auto(&a_smem)[2] = get_a_tile(s, input_stage);
                auto &b_smem = get_b_tile(s, input_stage);

                if (iter == 0) {
                    s.loader_record(WAIT_EVENT);
                    gmem_waiter::gmem_wait(g, s, inst);
                    s.loader_record(READY_EVENT);
                }

                if ((iter < INPUT_PIPELINE_STAGES) && (iter % 2 == 0)) {
                    s.wait_page_ready(a_page);  // Stall until A is ready.
                }

                // Load A
                int should_record = (iter < 8) || (iter >= _num_iters - 4);
                if (should_record) s.loader_record(LOAD_EVENT);
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    if constexpr (kittens::ducks::gl::all<std::remove_cvref_t<decltype(g.*A_Ptr)>>)
                        tma::load_async<_axis, cache_policy::NORMAL>(a_smem[i], g.*A_Ptr,
                                                                     {2 * inst.local_row + i, iter}, sem);
                    else
                        tma::load_async<_axis, cache_policy::NORMAL>(a_smem[i], (g.*A_Ptr)[g.dev_idx],
                                                                     {2 * inst.local_row + i, iter}, sem);
                }

                if (iter < INPUT_PIPELINE_STAGES) {
                    s.wait_page_ready(b_page);  // Stall until B is ready.
                }
                if (should_record) s.loader_record(LOAD2_EVENT);
                // Load B (always a local GL)
                tma::load_async<_axis, cache_policy::NORMAL>(b_smem, g.*B_Ptr, {layer_idx, inst.local_col, iter}, sem);

                // Advance
                input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
            }
        }
        static_assert(_num_iters >= INPUT_PIPELINE_STAGES, "Invalid number of iters");
    }

    template <int _pages_to_not_release = 0>
    __device__ static __forceinline__ rt_fl<16, 256> matmul_loop(state<Config> &s, const Globals &g) {
        static_assert(Config::NUM_CONSUMER_WARPS == 8);

        parsed_instruction inst{s};
        int groupid = warpgroup::groupid();

        int input_stage = 0;
        rt_fl<16, 256> out;

        {
            wait(inputs_arrived(s, input_stage), 0);
            auto(&a_smem)[2] = get_a_tile(s, input_stage);
            auto &b_smem = get_b_tile(s, input_stage);

            warpgroup::mm_ABt(out, a_smem[groupid], b_smem);
            s.consumer_record(COMPUTE_EVENT);
            warpgroup::mma_async_wait();
            warpgroup::arrive(inputs_finished(s, input_stage));

            input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
        }

        for (int i = 1; i < _num_iters; i++) {
            wait(inputs_arrived(s, input_stage), (i % (2 * INPUT_PIPELINE_STAGES)) >= INPUT_PIPELINE_STAGES);
            auto(&a_smem)[2] = get_a_tile(s, input_stage);
            auto &b_smem = get_b_tile(s, input_stage);

            warpgroup::mma_ABt(out, a_smem[groupid], b_smem);
            int should_record = (i < 8) || (i >= _num_iters - 4);
            if (should_record) s.consumer_record(COMPUTE_EVENT);

            warpgroup::mma_async_wait();
            warpgroup::arrive(inputs_finished(s, input_stage));

            if (i == _num_iters - 4) {
                if constexpr (_pages_to_not_release < 1) {
                    s.warp_finish_page(s.pid(1), 1);
                }
            } else if (i == _num_iters - 3) {
                if constexpr (_pages_to_not_release < 2) {
                    s.warp_finish_page(s.pid(0), 1);
                }
                if constexpr (_pages_to_not_release < 3) {
                    s.warp_finish_page(s.pid(2), 1);
                }
            } else if (i == _num_iters - 2) {
                if constexpr (_pages_to_not_release < 4) {
                    s.warp_finish_page(s.pid(4), 1);
                }
            } else if (i == _num_iters - 1) {
                if constexpr (_pages_to_not_release < 5) {
                    s.warp_finish_page(s.pid(3), 1);
                }
                if constexpr (_pages_to_not_release < 6) {
                    s.warp_finish_page(s.pid(5), 1);
                }
            }

            input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
        }

        return out;
    }
};
