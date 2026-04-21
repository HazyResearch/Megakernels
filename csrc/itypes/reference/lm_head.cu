#include "llama.cuh"
#include "matmul_pipeline.cuh"

using namespace kittens;
using namespace megakernel;

struct lm_head_gmem_waiter;

template <typename Config, typename Globals>
struct lm_head {
    static constexpr int opcode = OPCODE_LM_Head;
    static constexpr int NUM_ITERS = Globals::hidden_dim / PIPELINE_K_DIM;

    struct parsed_instruction {
        int layer_idx;  // unused
        int local_row;
        int local_col;
        int row;
        int col;
        __device__ inline parsed_instruction(typename Config::instruction_t &instruction) {
            layer_idx = instruction[1];
            local_row = instruction[2];
            local_col = instruction[3];
            row = instruction[4];
            col = instruction[5];
        }
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    using matmul_pipeline = matmul_pipeline<Config, Globals, parsed_instruction, lm_head_gmem_waiter,
                                            &Globals::rms_lm_head_intermediates, &Globals::lm_head_weights, NUM_ITERS>;

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query) {
            return matmul_pipeline::release_lid<2>(g, instruction, query);
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            return matmul_pipeline::init_semaphores(s);
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            matmul_pipeline::loader_loop(s, g);
            warp::arrive(s.instruction_fetch_ready, Config::NUM_CONSUMER_WARPS);
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {}
    };
    using half_consumer = group<Config::NUM_CONSUMER_WARPS / 2>;
    using constorer = group<Config::NUM_CONSUMER_WARPS + 1>;
    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            rt_bf<16, 256> out;
#ifdef KITTENS_BLACKWELL
            matmul_pipeline::matmul_loop<2>(s, g);
            wait(matmul_pipeline::outputs_arrived(s), 0);
            auto dt = s.tensor_alloc.template allocate<tt<float, 128, 256>>(half_consumer::groupid() * 256);
            half_consumer::load_async(out, dt);
            tensor_load_wait();
            __syncwarp();
            warp::arrive(s.tensor_finished);
#else
            rt_fl<16, 256> matmul_out;
            matmul_out = matmul_pipeline::matmul_loop<2>(s, g);
            warp::copy(out, matmul_out);
#endif
            s.consumer_record(STORE_EVENT);
            warpgroup::store(matmul_pipeline::get_output_tile(s, warpgroup::groupid()), out);
            int store_bar = 10 + s.instruction_index % 2;
            constorer::sync(store_bar);
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            int store_bar = 10 + s.instruction_index % 2;
            constorer::sync(store_bar);  // await arrive from consumer
            for (int i = 0; i < 2; i++) {
                s.storer_record(STORE_EVENT);
#ifdef KITTENS_BLACKWELL
                warp::tma::store_async(g.logits, matmul_pipeline::get_output_tile(s, i),
                                       {16 * inst.row + 8 * (i >= 8) + 2 * (i % 4) + ((i % 8) / 4), inst.col});
#else
                warp::tma::store_async(g.logits, matmul_pipeline::get_output_tile(s, i),
                                       {2 * inst.local_row + i, inst.local_col});
#endif
                tma::store_async_read_wait();
                s.storer_record(READY_EVENT);
                warp::sync();
                s.warp_finish_page(matmul_pipeline::get_output_page(s, i), Config::NUM_CONSUMER_WARPS);
            }
        }
    };
};

struct lm_head_gmem_waiter {
    template <typename Config, typename Globals, typename instruction_t>
    static __device__ inline void gmem_wait(const Globals &g, state<Config> &s, instruction_t &inst) {
        int expected_val = Globals::matmul_batch_block_size;
        wait_on_barrier<Scope::GPU>(&g.Bar[g.dev_idx][{0, OPCODE_LM_HeadNorm - 1, inst.local_row, 0}],
                        expected_val, "lm_head_gmem_waiter", "dev=%d, layer=0, row=%d",
                        g.dev_idx, inst.local_row);
    }
};
