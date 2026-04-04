#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "utils.cuh"
#include "itypes/llama1b/matvec_pipeline.cuh"

namespace megakittens {

template <typename Config, typename Globals, int N, int HEAD_DIM, int NUM_KV_HEADS,
          int SRC_ACT, int SRC_NORM, int SRC_QKV_W, int SRC_ROPE_COS, int SRC_ROPE_SIN,
          int SRC_K_CACHE, int SRC_V_CACHE, int DST_Q>
struct RmsQkvRopeAppend {

    static constexpr int BLOCK_SIZE = 16;
    static constexpr int K_BLK_START = N / BLOCK_SIZE;
    static constexpr int V_BLK_START = (N + NUM_KV_HEADS * HEAD_DIM) / BLOCK_SIZE;

    using rope_t = kittens::sv_fl<HEAD_DIM>;

    __device__ static inline uint8_t *get_rope_cos_ptr(state_t<Config> &s) {
        return reinterpret_cast<uint8_t *>(s.scratch());
    }
    __device__ static inline uint8_t *get_rope_sin_ptr(state_t<Config> &s) {
        return reinterpret_cast<uint8_t *>(s.scratch()) + sizeof(rope_t);
    }
    __device__ static inline rope_t &get_rope_cos(state_t<Config> &s) {
        return *reinterpret_cast<rope_t *>(get_rope_cos_ptr(s));
    }
    __device__ static inline rope_t &get_rope_sin(state_t<Config> &s) {
        return *reinterpret_cast<rope_t *>(get_rope_sin_ptr(s));
    }

    struct parsed_instruction {
        int layer_idx, start_block_idx, end_block_idx, iters;
        __device__ inline parsed_instruction(const instruction_t &instruction) {
            layer_idx       = instruction.indices[0];
            start_block_idx = instruction.indices[1];
            end_block_idx   = instruction.indices[2];
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

            // Load the 16 rope cos/sin values for this block directly from global memory
            int head_chunk = block_idx % (HEAD_DIM / BLOCK_SIZE);
            const float *cos_base = reinterpret_cast<const float *>(
                g.template gls<SRC_ROPE_COS>().raw_ptr)
                + static_cast<int>(g.pos_id) * HEAD_DIM + head_chunk * BLOCK_SIZE;
            const float *sin_base = reinterpret_cast<const float *>(
                g.template gls<SRC_ROPE_SIN>().raw_ptr)
                + static_cast<int>(g.pos_id) * HEAD_DIM + head_chunk * BLOCK_SIZE;

            kittens::rv_fl<16> rope_cos_rv, rope_sin_rv;
            if (kittens::laneid() < BLOCK_SIZE) {
                rope_cos_rv[0][0] = cos_base[kittens::laneid()];
                rope_sin_rv[0][0] = sin_base[kittens::laneid()];
            }

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
                        {inst.layer_idx, static_cast<int>(g.pos_id), head_idx, dim_idx});
                } else { // V
                    int base_index = (block_idx - V_BLK_START) * BLOCK_SIZE;
                    int head_idx = base_index / HEAD_DIM;
                    int dim_idx = (base_index % HEAD_DIM) / BLOCK_SIZE;
                    kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                        g.template gls<SRC_V_CACHE>(), qkv_proj_smem_bf,
                        {inst.layer_idx, static_cast<int>(g.pos_id), head_idx, dim_idx});
                }

                kittens::tma::store_async_read_wait();
                // TODO: reference does store_async_wait() (full, not read) here
                // current is batch-signal at end of storer via all_barrier_arrive.
            }

            kittens::warp::sync();
        }
    };

    using pipeline = llama1b::rms_matvec_pipeline<
        Config, Globals, N, parsed_instruction, pipeline_specifics, SRC_ACT, SRC_NORM>;

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
            if (kittens::laneid() == 0) {
                __threadfence_block();
                kittens::arrive(rope_arrived(s));
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
            pipeline::storer_loop(s, g);
            if (kittens::warp::elect_leader()) {
                __threadfence();
                all_barrier_arrive<Config>(g, s.instruction());
            }
        }
    };
};

} // namespace megakittens
