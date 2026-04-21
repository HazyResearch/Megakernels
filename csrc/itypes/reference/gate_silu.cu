#include "llama.cuh"
#include "matmul_pipeline.cuh"

using namespace kittens;
using namespace megakernel;

struct gate_silu_gmem_waiter;

template <typename Config, typename Globals>
struct gate_silu {
    static constexpr int opcode = OPCODE_GateSiLU;
    static constexpr int prev_opcode = opcode - 1;
    static constexpr int NUM_ITERS = Globals::hidden_dim / PIPELINE_K_DIM;

    struct parsed_instruction {
        int layer;
        int local_row;
        int local_col;
        int row;
        int col;
        __device__ inline parsed_instruction(typename Config::instruction_t &instruction) {
            layer = instruction[1];
            local_row = instruction[2];
            local_col = instruction[3];
            row = instruction[4];
            col = instruction[5];
        }
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    using matmul_pipeline = matmul_pipeline<Config, Globals, parsed_instruction, gate_silu_gmem_waiter,
                                            &Globals::rms_gate_intermediates, &Globals::gate_weights, NUM_ITERS>;

    __device__ static inline semaphore &output_arrived(state<Config> &s) {
        return s.semaphores()[matmul_pipeline::SEM_COUNT];
    }

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query) {
            return matmul_pipeline::release_lid<2>(g, instruction, query);
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            if(laneid() == 0) init_semaphore(output_arrived(s), Config::NUM_CONSUMER_WARPS);
            return matmul_pipeline::init_semaphores(s) + 1;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            matmul_pipeline::loader_loop<true>(s, g, inst.layer);
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
            rt_fl<16, 256> out_fl;
#ifdef KITTENS_BLACKWELL
            matmul_pipeline::matmul_loop<2>(s, g);
            wait(matmul_pipeline::outputs_arrived(s), 0);
            auto dt = s.tensor_alloc.template allocate<tt<float, 128, 256>>(half_consumer::groupid() * 256);
            half_consumer::load_async(out_fl, dt);
            tensor_load_wait();
#else
            out_fl = matmul_pipeline::matmul_loop<2>(s, g);
#endif

            rt_fl<16, 256> gate_buf;
            half_consumer::copy(gate_buf, out_fl);
            half_consumer::mul(gate_buf, gate_buf, -1);
            half_consumer::exp(gate_buf, gate_buf);
            half_consumer::add(gate_buf, gate_buf, 1);
            half_consumer::div(out_fl, out_fl, gate_buf);

            __syncwarp();
#ifdef KITTENS_BLACKWELL
            warp::arrive(s.tensor_finished);
#endif
            s.consumer_record(STORE_EVENT);
            warpgroup::store(matmul_pipeline::get_output_tile(s, warpgroup::groupid()), out_fl);
            warp::sync();
            warp::arrive(output_arrived(s));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            wait(output_arrived(s), 0);
            for (int i = 0; i < 2; i++) {
                s.storer_record(STORE_EVENT);
#ifdef KITTENS_BLACKWELL
                warp::tma::store_async(
                    g.silu_out, matmul_pipeline::get_output_tile(s, i),
                    {16 * inst.row + 8 * (i >= 8) + 2 * (i % 4) + ((i % 8) / 4), (g.dev_idx * 14) + inst.col});
#else
                warp::tma::store_async(g.silu_out, matmul_pipeline::get_output_tile(s, i),
                                       {2 * inst.local_row + i, inst.local_col});
#endif
                tma::store_async_read_wait();
                s.storer_record(READY_EVENT);
                warp::sync();
                s.warp_finish_page(matmul_pipeline::get_output_page(s, i), Config::NUM_CONSUMER_WARPS);
            }
            tma::store_async_wait();
            warp::sync();
            fence<Sem::RELEASE, Scope::GPU>();
            s.storer_record(STORE2_EVENT);
            if (kittens::laneid() == 0) {
                redAdd<Sem::RELAXED, Scope::GPU>(&g.Bar[g.dev_idx][{inst.layer, opcode - 1, inst.local_row, inst.local_col}], 1);
            }
        }
    };
};

struct gate_silu_gmem_waiter {
    template <typename Config, typename Globals, typename instruction_t>
    static __device__ inline void gmem_wait(const Globals &g, state<Config> &s, instruction_t &inst) {
        int expected_val = Globals::matmul_batch_block_size;
        wait_on_barrier<Scope::SYS>(&g.Bar[g.dev_idx][{inst.layer, OPCODE_MlpNorm - 1, inst.row, 0}],
                        expected_val, "gate_silu_gmem_waiter", "dev=%d, layer=%d, row=%d",
                        g.dev_idx, inst.layer, inst.row);
    }
};
