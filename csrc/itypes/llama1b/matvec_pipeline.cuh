#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "utils.cuh"
#include "itypes/llama1b/utils.cuh"

namespace megakittens {
namespace llama1b {

template <typename Config, typename Globals, int N,
          typename parsed_instruction, typename pipeline_specifics>
struct matvec_pipeline {
    static constexpr int INPUT_PIPELINE_STAGES = 3;
    static constexpr int OUTPUT_PIPELINE_STAGES = 3;
    static constexpr int STAGE_PAGES = 2;
    static constexpr int ACTIVATION_PAGE = 0;
    static constexpr int WEIGHTS_START_PAGE = 1;

    static constexpr int MATVEC_BLOCK_SIZE = 16;
    static constexpr int TILES_PER_PAGE = Config::NUM_CONSUMER_WARPS / STAGE_PAGES; // 4
    static constexpr int TILES_PER_STAGE = STAGE_PAGES * TILES_PER_PAGE;             // 8

    static constexpr int REDUCTION_DIM_PER_WARP = N / Config::NUM_CONSUMER_WARPS;
    static constexpr int WARPS_PER_PAGE = Config::NUM_CONSUMER_WARPS / STAGE_PAGES;

    static constexpr int SEM_COUNT = 1 + (INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES) * 2;

    static constexpr int SCRATCH_BYTES_PER_WARP = MATVEC_BLOCK_SIZE * sizeof(float);
    static constexpr int SCRATCH_BYTES_PER_STAGE = SCRATCH_BYTES_PER_WARP * Config::NUM_CONSUMER_WARPS;

    // offsets on activation page
    static constexpr int OUTPUT_SCRATCH_OFFSET = N * sizeof(kittens::bf16);

    __device__ static inline kittens::semaphore &activations_arrived(state_t<Config> &s) {
        return s.semaphores()[0];
    }
    __device__ static inline kittens::semaphore &weights_arrived(state_t<Config> &s, int stage) {
        return s.semaphores()[1 + stage];
    }
    __device__ static inline kittens::semaphore &weights_finished(state_t<Config> &s, int stage) {
        return s.semaphores()[1 + INPUT_PIPELINE_STAGES + stage];
    }
    __device__ static inline kittens::semaphore &outputs_arrived(state_t<Config> &s, int stage) {
        return s.semaphores()[1 + 2 * INPUT_PIPELINE_STAGES + stage];
    }
    __device__ static inline kittens::semaphore &outputs_finished(state_t<Config> &s, int stage) {
        return s.semaphores()[1 + 2 * INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES + stage];
    }

    __device__ static inline int get_activation_page(state_t<Config> &s) {
        return s.lid_to_pid(ACTIVATION_PAGE);
    }
    __device__ static inline int get_weight_page(state_t<Config> &s, int stage, int page) {
        return s.lid_to_pid(WEIGHTS_START_PAGE + stage * STAGE_PAGES + page);
    }

    __device__ static inline kittens::sv_bf<N> &get_activations(state_t<Config> &s) {
        return s.pages[get_activation_page(s)].template as<kittens::sv_bf<N>>();
    }
    __device__ static inline uint8_t *get_output_start(state_t<Config> &s, int stage) {
        return reinterpret_cast<uint8_t *>(
            s.pages[get_activation_page(s)].ptr(OUTPUT_SCRATCH_OFFSET + stage * SCRATCH_BYTES_PER_STAGE));
    }


    __device__ static inline int
    lid_release_order(const Globals &g, state_t<Config> &s, int query) {
        // last query is always activation page because our scratch is inside it
        if (query == Config::NUM_PAGES - 1) return ACTIVATION_PAGE;

        parsed_instruction inst{s.instruction()};
        int remainder = inst.iters % INPUT_PIPELINE_STAGES;

        if (inst.iters == 1) {
            // unused pages first
            constexpr int order[] = {3, 4, 5, 6, 1, 2};
            return order[query];
        } else if (inst.iters == 2) {
            constexpr int order[] = {5, 6, 1, 2, 3, 4};
            return order[query];
        } else if (remainder == 1) {
            // 3 and 4 finish first.
            constexpr int order[] = {3, 4, 5, 6, 1, 2};
            return order[query];
        } else if (remainder == 2) {
            constexpr int order[] = {5, 6, 1, 2, 3, 4};
            return order[query];
        } else {
            constexpr int order[] = {1, 2, 3, 4, 5, 6};
            return order[query];
        }
    }

    __device__ static inline int init_semaphores(const Globals &g, state_t<Config> &s) {
        // one lane inits each semaphore, consult gemm.cuh pattern
        if (kittens::laneid() == 0)
            kittens::init_semaphore(activations_arrived(s), 1);
        if (kittens::laneid() < INPUT_PIPELINE_STAGES)
            kittens::init_semaphore(weights_arrived(s, kittens::laneid()), 1);
        if (kittens::laneid() < INPUT_PIPELINE_STAGES)
            kittens::init_semaphore(weights_finished(s, kittens::laneid()), Config::NUM_CONSUMER_WARPS);
        if (kittens::laneid() < OUTPUT_PIPELINE_STAGES)
            kittens::init_semaphore(outputs_arrived(s, kittens::laneid()), Config::NUM_CONSUMER_WARPS);
        if (kittens::laneid() < OUTPUT_PIPELINE_STAGES)
            kittens::init_semaphore(outputs_finished(s, kittens::laneid()), 1);
        return SEM_COUNT;
    }


    __device__ static inline void loader_loop(state_t<Config> &s, 
                                              const Globals &g) {
        parsed_instruction inst{s.instruction()};

        int needed_pages = 
            1 + min(inst.iters, INPUT_PIPELINE_STAGES) * STAGE_PAGES;

        if (kittens::laneid() == 0) {
            s.page_wait(get_activation_page(s));

            int input_stage = 0;
            for (int iter = 0; iter < inst.iters; iter++) {
                kittens::wait(weights_finished(s, input_stage),
                    (iter % (2 * INPUT_PIPELINE_STAGES)) < INPUT_PIPELINE_STAGES);

                auto &sem = weights_arrived(s, input_stage);
                kittens::tma::expect_bytes(sem, sizeof(kittens::bf16) * N * MATVEC_BLOCK_SIZE);

                // 2 pages per stage, 2 × st_bf<16,512> per page = 4 TMA loads
                #pragma unroll
                for (int i = 0; i < STAGE_PAGES * 2; i++) {
                    int pid = get_weight_page(s, input_stage, i / 2);
                    if (iter < INPUT_PIPELINE_STAGES && i % 2 == 0)
                        s.page_wait(pid);
                    auto &tile = s.pages[pid].template as<
                        kittens::st_bf<MATVEC_BLOCK_SIZE, 512>>(
                        (i % 2) * sizeof(kittens::st_bf<MATVEC_BLOCK_SIZE, 512>));
                    pipeline_specifics::load_iter(s, g, inst, iter, i, tile, sem);
                }
                input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
            }
        }

        if (kittens::laneid() >= needed_pages && kittens::laneid() < Config::NUM_PAGES) {
            int pid = s.lid_to_pid(kittens::laneid());
            s.page_wait(pid);
            s.page_finish(pid);
        }
    }

    template <int output_scratch_off = OUTPUT_SCRATCH_OFFSET, typename rv_t>
    __device__ static inline void
    consumer_loop(state_t<Config> &s, const Globals &g, rv_t &activations_vec) {
        
        parsed_instruction inst{s.instruction()};

        int page_index   = kittens::warpid() / WARPS_PER_PAGE;

        int activation_page      = get_activation_page(s);

        int input_stage = 0, output_stage = 0;
        for (int i = 0; i < inst.iters; i++) {
            int weight_page = get_weight_page(s, input_stage, page_index);
            kittens::wait(weights_arrived(s, input_stage),
                (i % (2 * INPUT_PIPELINE_STAGES)) >= INPUT_PIPELINE_STAGES);
            kittens::wait(outputs_finished(s, output_stage),
                (i % (2 * OUTPUT_PIPELINE_STAGES)) < OUTPUT_PIPELINE_STAGES);
            kittens::st_bf<MATVEC_BLOCK_SIZE, REDUCTION_DIM_PER_WARP> &weights =
                reinterpret_cast<kittens::st_bf<MATVEC_BLOCK_SIZE, REDUCTION_DIM_PER_WARP> *>(
                    s.pages[weight_page].ptr())[kittens::warpid() % WARPS_PER_PAGE];

            // output scratch is on the activation page (COULD BE CHANGED)
            uint8_t *output_scratch_start = reinterpret_cast<uint8_t *>(
                s.pages[activation_page].ptr(
                    output_scratch_off + output_stage * SCRATCH_BYTES_PER_STAGE));
            kittens::sv_fl<MATVEC_BLOCK_SIZE> &out_smem =
                *reinterpret_cast<kittens::sv_fl<MATVEC_BLOCK_SIZE> *>(
                    output_scratch_start + kittens::warpid() * SCRATCH_BYTES_PER_WARP);

            llama1b::matvec(out_smem, weights, activations_vec);

            kittens::warp::sync();
            kittens::warp::arrive(outputs_arrived(s, output_stage));
            kittens::warp::arrive(weights_finished(s, input_stage));

            input_stage  = (input_stage  + 1) % INPUT_PIPELINE_STAGES;
            output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
        }

        // release after consumer loop. could do in between but i was having issues
        kittens::group<Config::NUM_CONSUMER_WARPS>::sync(1);
        if (kittens::warpid() == 0 && kittens::warp::elect_leader()) {
            int used_stages = min(inst.iters, INPUT_PIPELINE_STAGES);
            for (int stage = 0; stage < used_stages; stage++)
                for (int p = 0; p < STAGE_PAGES; p++)
                    s.page_finish(get_weight_page(s, stage, p));
        }
    }

    template <int iter_scale = 1>
    __device__ static inline void storer_loop(state_t<Config> &s, const Globals &g) {
        parsed_instruction inst{s.instruction()};
        
        int output_stage = 0;
        for (int i = 0; i < inst.iters; i++) {
            kittens::wait(outputs_arrived(s, output_stage),
                (i % (2 * OUTPUT_PIPELINE_STAGES)) >= OUTPUT_PIPELINE_STAGES);

            pipeline_specifics::store(s, g, inst, i, output_stage);

            // time here

            if ((i + 1) % iter_scale == 0) {
                for (int j = 0; j < iter_scale; j++) {
                    int stage = (output_stage - j + OUTPUT_PIPELINE_STAGES) % OUTPUT_PIPELINE_STAGES;
                    kittens::warp::arrive(outputs_finished(s, stage));
                }
            }
            output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
        }

        // storer is the last to read activation page. so it releases
        kittens::warp::sync();
        if (kittens::warp::elect_leader()) {
            kittens::tma::store_async_wait();
            s.page_finish(get_activation_page(s));
        }
    }
};

template <typename Config, typename Globals, int N,
          typename parsed_instruction, typename pipeline_specifics,
          int SRC_ACT, int SRC_NORM>
struct rms_matvec_pipeline
    : public matvec_pipeline<Config, Globals, N, parsed_instruction, pipeline_specifics> {

    using pipeline = matvec_pipeline<Config, Globals, N, parsed_instruction, pipeline_specifics>;

    // Activation page layout (with RMS scale):
    static constexpr int ACTIVATIONS_SIZE      = N * sizeof(kittens::bf16);
    static constexpr int RMS_SCALE_OFFSET      = ACTIVATIONS_SIZE;
    static constexpr int RMS_SCALE_SIZE        = N * sizeof(kittens::bf16);
    static constexpr int RMS_SCRATCH_OFFSET    = RMS_SCALE_OFFSET + RMS_SCALE_SIZE;
    static constexpr int RMS_SCRATCH_SIZE      = Config::NUM_CONSUMER_WARPS * sizeof(float);
    // Output scratch must be 1024-byte aligned for TMA
    static constexpr int OUTPUT_SCRATCH_OFFSET = ((RMS_SCRATCH_OFFSET + RMS_SCRATCH_SIZE) + 1023) & ~1023;

    // +1 semaphore for rms_scale_arrived (matches original)
    static constexpr int SEM_COUNT = pipeline::SEM_COUNT + 1;

    __device__ static inline kittens::semaphore &rms_scale_arrived(state_t<Config> &s) {
        return s.semaphores()[pipeline::SEM_COUNT];
    }
    __device__ static inline kittens::sv_bf<N> &get_rms_scale(state_t<Config> &s) {
        return s.pages[pipeline::get_activation_page(s)].template as<kittens::sv_bf<N>>(RMS_SCALE_OFFSET);
    }
    __device__ static inline uint8_t *get_output_start(state_t<Config> &s, int stage) {
        return reinterpret_cast<uint8_t *>(
            s.pages[pipeline::get_activation_page(s)].ptr(
                OUTPUT_SCRATCH_OFFSET + stage * pipeline::SCRATCH_BYTES_PER_STAGE));
    }

    __device__ static inline int
    lid_release_order(const Globals &g, state_t<Config> &s, int query) {
        return pipeline::lid_release_order(g, s, query);
    }

    __device__ static inline int init_semaphores(const Globals &g, state_t<Config> &s) {
        pipeline::init_semaphores(g, s);
        if (kittens::laneid() == 0)
            kittens::init_semaphore(rms_scale_arrived(s), 1);
        return SEM_COUNT;
    }

    __device__ static inline void loader_loop(state_t<Config> &s, const Globals &g) {
        if (kittens::laneid() == 0) {
            s.page_wait(pipeline::get_activation_page(s));

            // Wait for upstream (e.g. last downproj)
            pipeline_specifics::gmem_wait(g, s);

            // TMA load hidden_states and RMS scale into activation page
            auto &activations = pipeline::get_activations(s);
            auto &rms_scale   = get_rms_scale(s);

            auto &act_sem = pipeline::activations_arrived(s);
            kittens::tma::expect_bytes(act_sem, sizeof(activations));
            kittens::tma::load_async(activations, g.template gls<SRC_ACT>(), {0, 0}, act_sem);

            auto &rms_sem = rms_scale_arrived(s);
            kittens::tma::expect_bytes(rms_sem, sizeof(rms_scale));
            kittens::tma::load_async(rms_scale, g.template gls<SRC_NORM>(), {0, 0}, rms_sem);
        }
        pipeline::loader_loop(s, g);
    }

    __device__ static inline void consumer_loop(state_t<Config> &s, const Globals &g) {
        constexpr int ELEMS_PER_WARP = N / Config::NUM_CONSUMER_WARPS;
        using sv_slice_t = kittens::sv_bf<ELEMS_PER_WARP>;

        kittens::wait(pipeline::activations_arrived(s), 0);
        kittens::wait(rms_scale_arrived(s), 0);

        auto &rms_scale_smem   = reinterpret_cast<sv_slice_t *>(&get_rms_scale(s))[kittens::warpid()];
        auto &activations_smem = reinterpret_cast<sv_slice_t *>(&pipeline::get_activations(s))[kittens::warpid()];

        float *rms_scratch = reinterpret_cast<float *>(
            s.pages[pipeline::get_activation_page(s)].ptr(RMS_SCRATCH_OFFSET));
        auto activations_vec = llama1b::rms_norm<Config, N>(
            rms_scale_smem, activations_smem, g.rms_norm_eps, rms_scratch);

        kittens::warp::sync();

        pipeline::template consumer_loop<OUTPUT_SCRATCH_OFFSET>(s, g, activations_vec);
    }
};

} // namespace llama1b
} // namespace megakittens
