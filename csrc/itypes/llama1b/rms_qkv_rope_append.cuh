#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "utils.cuh"
#include "itypes/llama1b/matvec_pipeline.cuh"

namespace megakittens {

template <typename Config, typename Globals, int N, int HEAD_DIM, int NUM_KV_HEADS,
          int SRC_ACT, int SRC_NORM, int SRC_QKV_W, int SRC_ROPE_COS, int SRC_ROPE_SIN,
          int SRC_K_CACHE, int SRC_V_CACHE, int SCALAR_POS_ID, int SCALAR_RMS_EPS, int DST_Q>
struct RmsQkvRopeAppend {

    static constexpr int BLOCK_SIZE = 16;
    static constexpr int K_BLK_START = N / BLOCK_SIZE;
    static constexpr int V_BLK_START = (N + NUM_KV_HEADS * HEAD_DIM) / BLOCK_SIZE;

    using rope_t = kittens::sv_fl<HEAD_DIM>;

    struct parsed_instruction {
        int layer_idx, start_block_idx, end_block_idx, iters, barrier_base;
        __device__ inline parsed_instruction(const instruction_t &instruction) {
            layer_idx       = instruction.indices[0];
            start_block_idx = instruction.indices[1];
            end_block_idx   = instruction.indices[2];
            barrier_base    = instruction.indices[3];
            iters           = end_block_idx - start_block_idx;
        }
        __device__ inline parsed_instruction(state_t<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        __device__ static inline void gmem_wait(const Globals &g, state_t<Config> &s) {
            all_input_barrier_wait<Config>(g, s.instruction());
        }

        __device__ static inline void
        load_iter(state_t<Config> &s, const Globals &g, parsed_instruction &inst,
                  int iter, int col_idx,
                  kittens::st_bf<16, 512> &weight_chunk,
                  kittens::semaphore &sem) {
            int block_idx = inst.start_block_idx + iter;
            kittens::tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                weight_chunk, g.template gls<SRC_QKV_W>(),
                {inst.layer_idx, block_idx, col_idx}, sem);
        }

        __device__ static inline void
        store(state_t<Config> &s, const Globals &g, parsed_instruction &inst,
              int output_idx, int output_stage) {
            int block_idx = inst.start_block_idx + output_idx;

            uint8_t *output_scratch = pipeline::get_output_start(s, output_stage);

            kittens::sv_bf<16> &qkv_proj_smem_bf =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch);

            kittens::rv_fl<16> qkv_proj;

            llama1b::matvec_reduce<Config, pipeline::SCRATCH_BYTES_PER_WARP>(
                output_scratch, qkv_proj);

            kittens::rv_fl<16> rope_cos_rv, rope_sin_rv;

            kittens::wait(rope_arrived(s), 0);

            auto head_chunk = block_idx % (HEAD_DIM / BLOCK_SIZE);

            kittens::sv_fl<16> &rope_cos_sv = *reinterpret_cast<kittens::sv_fl<16> *>(
                get_rope_cos_ptr(s) + head_chunk * BLOCK_SIZE * sizeof(float));
            kittens::sv_fl<16> &rope_sin_sv = *reinterpret_cast<kittens::sv_fl<16> *>(
                get_rope_sin_ptr(s) + head_chunk * BLOCK_SIZE * sizeof(float));

            kittens::warp::load(rope_cos_rv, rope_cos_sv);
            kittens::warp::load(rope_sin_rv, rope_sin_sv);

            if (block_idx < V_BLK_START) {

                int mod = (kittens::laneid() & 0b1) ? -1 : 1; // 1 for even, -1 for odd
                kittens::warp::sync();
                float pair_val =
                    __shfl_sync(0xFFFFFFFF, qkv_proj[0][0], kittens::laneid() + mod);

                if (kittens::laneid() < 16) {
                    qkv_proj[0][0] =
                        float(qkv_proj[0][0]) * rope_cos_rv[0][0] +
                        float(-1 * mod) * float(pair_val) * rope_sin_rv[0][0];
                }
            }

            kittens::warp::sync();
            kittens::warp::store(qkv_proj_smem_bf, qkv_proj);
            kittens::warp::sync();

            if (kittens::warp::elect_leader()) {

                if (block_idx < K_BLK_START) { // Q
                    kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                        g.template gls<DST_Q>(), qkv_proj_smem_bf, {0, block_idx});
                } else if (block_idx < V_BLK_START) { // K
                    int base_index = (block_idx - K_BLK_START) * BLOCK_SIZE;
                    int head_idx = base_index / HEAD_DIM;
                    int dim_idx = (base_index % HEAD_DIM) / BLOCK_SIZE;
                    kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                        g.template gls<SRC_K_CACHE>(), qkv_proj_smem_bf,
                        {inst.layer_idx, static_cast<int>(g.template gls<SCALAR_POS_ID>().raw_ptr[0]), head_idx, dim_idx});
                } else { // V
                    int base_index = (block_idx - V_BLK_START) * BLOCK_SIZE;
                    int head_idx = base_index / HEAD_DIM;
                    int dim_idx = (base_index % HEAD_DIM) / BLOCK_SIZE;
                    kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                        g.template gls<SRC_V_CACHE>(), qkv_proj_smem_bf,
                        {inst.layer_idx, static_cast<int>(g.template gls<SCALAR_POS_ID>().raw_ptr[0]), head_idx, dim_idx});
                }

                kittens::tma::store_async_wait();

                // Per-block fine-grained barrier: block_idx / (HEAD_DIM/BLOCK_SIZE)
                // maps Q blocks to Q head groups, K blocks to K head groups,
                // V blocks to V head groups — matching reference's block_idx/4.
                barrier_arrive<Config>(
                    &g.barriers.raw_ptr[inst.barrier_base + block_idx / (HEAD_DIM / BLOCK_SIZE)], 1);
            }

            kittens::warp::sync();
        }
    };

    using pipeline = llama1b::rms_matvec_pipeline<
        Config, Globals, N, parsed_instruction, pipeline_specifics, SRC_ACT, SRC_NORM, SCALAR_RMS_EPS>;

    static constexpr int ROPE_COS_OFFSET = ((pipeline::OUTPUT_SCRATCH_OFFSET + // 1024-align
        pipeline::OUTPUT_PIPELINE_STAGES * pipeline::SCRATCH_BYTES_PER_STAGE) + 1023) & ~1023;
    static constexpr int ROPE_SIN_OFFSET = ROPE_COS_OFFSET + HEAD_DIM * sizeof(float);

    __device__ static inline uint8_t *get_rope_cos_ptr(state_t<Config> &s) {
        return reinterpret_cast<uint8_t *>(
            s.pages[pipeline::get_activation_page(s)].ptr(ROPE_COS_OFFSET));
    }
    __device__ static inline uint8_t *get_rope_sin_ptr(state_t<Config> &s) {
        return reinterpret_cast<uint8_t *>(
            s.pages[pipeline::get_activation_page(s)].ptr(ROPE_SIN_OFFSET));
    }

    __device__ static inline kittens::semaphore &rope_arrived(state_t<Config> &s) {
        return s.semaphores()[pipeline::SEM_COUNT];
    }

    struct controller {
        __device__ __forceinline__ static int
        lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            return pipeline::lid_release_order(g, s, query);
        }
        __device__ __forceinline__ static int
        init_semaphores(const Globals &g, state_t<Config> &s) {
            pipeline::init_semaphores(g, s);
            if (kittens::warp::elect_leader())
                kittens::init_semaphore(rope_arrived(s), 1);
            return pipeline::SEM_COUNT + 1;
        }
    };

    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            if (kittens::warp::elect_leader()) {
                s.page_wait(pipeline::get_activation_page(s));

                rope_t &rope_cos_smem = *reinterpret_cast<rope_t *>(get_rope_cos_ptr(s));
                rope_t &rope_sin_smem = *reinterpret_cast<rope_t *>(get_rope_sin_ptr(s));
                auto &sem = rope_arrived(s);
                kittens::tma::expect_bytes(sem, 2 * HEAD_DIM * sizeof(float));
                kittens::tma::load_async<kittens::cache_policy::EVICT_LAST>(
                    rope_cos_smem, g.template gls<SRC_ROPE_COS>(),
                    {0, 0, static_cast<int>(g.template gls<SCALAR_POS_ID>().raw_ptr[0]), 0}, sem);
                kittens::tma::load_async<kittens::cache_policy::EVICT_LAST>(
                    rope_sin_smem, g.template gls<SRC_ROPE_SIN>(),
                    {0, 0, static_cast<int>(g.template gls<SCALAR_POS_ID>().raw_ptr[0]), 0}, sem);
            }

            parsed_instruction inst{s.instruction()};
            pipeline::loader_loop(s, g, inst.layer_idx);
        }
    };

    struct launcher {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::launcher_loop(s, g);
        }
    };

    struct consumer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::consumer_loop(s, g);
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            // Per-block barrier signaling happens inside store().
            // num_dst_barriers=0 so all_barrier_arrive is not needed.
            pipeline::storer_loop(s, g);
        }
    };
};

} // namespace megakittens
