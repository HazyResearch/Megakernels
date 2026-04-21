#include "llama.cuh"

using namespace kittens;
using namespace megakernel;

template <int _opcode, typename Config, typename Globals>
struct barrier_inc_op {
    static constexpr int opcode = _opcode;
    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query) {
            return query;
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) { return 0; }
    };
    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            warp::arrive(s.instruction_fetch_ready, Config::NUM_CONSUMER_WARPS);
            if (laneid() < Config::NUM_PAGES) {
                auto pid = s.pid(laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
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
        static __device__ void run(const Globals &g, state<Config> &s) {}
    };

    struct storer {
        // Uses 4 full pages for outputs.
        static __device__ void run(const Globals &g, state<Config> &s) {
            constexpr int MATMUL_BATCH_HEIGHT = Globals::matmul_batch_block_size;
            int batch_size = g.batch_size;
            // int batch_size_wave_remainder = batch_size % (g.num_devices*MATMUL_BATCH_HEIGHT);
            // int batch_size_gpu_remainder = batch_size_wave_remainder % MATMUL_BATCH_HEIGHT;
            int batch_size_ceil = (batch_size + MATMUL_BATCH_HEIGHT - 1) & ~(MATMUL_BATCH_HEIGHT - 1);
            int extra_batch_size = batch_size_ceil - batch_size;
            if (extra_batch_size > 0) {
                int global_batch_row = batch_size / MATMUL_BATCH_HEIGHT;
                (void)global_batch_row;
                for (int i = laneid(); i < g.qkv_weights.depth(); i += kittens::WARP_THREADS) {
                    redAdd<Sem::RELAXED, Scope::SYS>(&g.Bar[g.dev_idx][{i, OPCODE_AttnNorm - 1, global_batch_row, 0}], extra_batch_size);
                    redAdd<Sem::RELAXED, Scope::SYS>(&g.Bar[g.dev_idx][{i, OPCODE_MlpNorm - 1, global_batch_row, 0}], extra_batch_size);
                }
                int2 local_block_idx = global_batch_idx_to_local_idx_info(g, batch_size);
                if (local_block_idx.x == g.dev_idx) {
                    for (int i = laneid(); i < g.qkv_weights.depth(); i += kittens::WARP_THREADS) {
                        redAdd<Sem::RELAXED, Scope::SYS>(&g.Bar[g.dev_idx][{i, OPCODE_GQA_AttentionDecode - 1,
                                                     local_block_idx.y / MATMUL_BATCH_HEIGHT, 0}],
                                  extra_batch_size * Globals::num_attention_heads);
                    }
                    if (laneid() == 0)
                        redAdd<Sem::RELAXED, Scope::SYS>(
                            &g.Bar[g.dev_idx][{0, OPCODE_LM_HeadNorm - 1, local_block_idx.y / MATMUL_BATCH_HEIGHT, 0}],
                            extra_batch_size);
                }
            }
        }
    };
};

template <typename Config, typename Globals>
struct barrier_inc : barrier_inc_op<OPCODE_Barrier_Inc, Config, Globals> {};
