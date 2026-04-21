#include "llama.cuh"
#include "matmul_pipeline.cuh"

using namespace kittens;
using namespace megakernel;

struct up_matmul_gmem_waiter;

template <typename Config, typename Globals>
struct up_matmul {
    static constexpr int opcode = OPCODE_UpMatmul;
    static constexpr int prev_opcode = opcode - 1;
    static constexpr int NUM_ITERS = Globals::hidden_dim / PIPELINE_K_DIM;

#ifdef KITTENS_BLACKWELL
    using silu_tile = st_bf<128, 256>;
#else
    using silu_tile = st_bf<64, 256>;
#endif

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

    using matmul_pipeline = matmul_pipeline<Config, Globals, parsed_instruction, up_matmul_gmem_waiter,
                                            &Globals::rms_gate_intermediates, &Globals::up_weights, NUM_ITERS>;

    __device__ static inline semaphore &silu_arrived(state<Config> &s, int laneid) {
        return s.semaphores()[matmul_pipeline::SEM_COUNT + laneid];
    }
    __device__ static inline semaphore &outputs_arrived(state<Config> &s) {
        return s.semaphores()[matmul_pipeline::SEM_COUNT + 2];
    }

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query) {
            return matmul_pipeline::release_lid<2>(g, instruction, query);
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            if(laneid()  < 2) init_semaphore(silu_arrived(s, laneid()), 1);
            if(laneid() == 0) init_semaphore(outputs_arrived(s), Config::NUM_CONSUMER_WARPS);
            return matmul_pipeline::init_semaphores(s) + 3;  // +2 for silu_arrived
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};

            // Once this loop is done, all pages used are released and ready for reuse.
            matmul_pipeline::loader_loop<true>(s, g, inst.layer);
            warp::sync();  // need to sync here
            if (warp::laneid() < 2) {
                int last_stage = (NUM_ITERS - 1) % matmul_pipeline::INPUT_PIPELINE_STAGES;
                wait(matmul_pipeline::inputs_finished(s, last_stage),
                     ((NUM_ITERS - 1) % (2 * matmul_pipeline::INPUT_PIPELINE_STAGES)) >=
                         matmul_pipeline::INPUT_PIPELINE_STAGES);
                int unfreed_page_base = matmul_pipeline::get_output_page(s, warp::laneid());
                silu_tile &silu_out = *reinterpret_cast<silu_tile *>(s.pages[unfreed_page_base].data);
                wait_on_barrier<Scope::GPU>(&g.Bar[g.dev_idx][{inst.layer, OPCODE_GateSiLU - 1, inst.local_row, inst.local_col}],
                                1, "loader GateSiLU barrier", "dev=%d, layer=%d, row=%d, col=%d",
                                g.dev_idx, inst.layer, inst.local_row, inst.local_col);
                s.loader_record(LOAD2_EVENT);
                tma::expect(silu_arrived(s, warp::laneid()), silu_out);
                tma::load_async(silu_out, g.silu_out, {(inst.local_row * 2) + warp::laneid(), inst.local_col},
                                silu_arrived(s, warp::laneid()));
            }
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
            out_fl = matmul_pipeline::template matmul_loop<2>(s, g);
#endif
            rt_fl<16, 256> silu_fl;

            // Load gate_silu intermediates here
            wait(silu_arrived(s, half_consumer::groupid()), 0);
            s.consumer_record(READY_EVENT);

            int unfreed_page_base = matmul_pipeline::get_output_page(s, half_consumer::groupid());
            silu_tile &silu_out = *reinterpret_cast<silu_tile *>(s.pages[unfreed_page_base].data);
            half_consumer::load(silu_fl, silu_out);

            half_consumer::sync(half_consumer::groupid() + 3);

            warp::mul(out_fl, out_fl, silu_fl);

#ifdef KITTENS_BLACKWELL
            warp::arrive(s.tensor_finished);
#endif
            s.consumer_record(STORE_EVENT);
            warpgroup::store(matmul_pipeline::get_output_tile(s, warpgroup::groupid()), out_fl);
            warp::sync();
            warp::arrive(outputs_arrived(s));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            wait(outputs_arrived(s), 0);
            for (int i = 0; i < 2; i++) {
                s.storer_record(STORE_EVENT);
#ifdef KITTENS_BLACKWELL
                warp::tma::store_async(g.silu_out, matmul_pipeline::get_output_tile(s, i),
                                       {16 * inst.row + 8 * (i >= 8) + 2 * (i % 4) + ((i % 8) / 4), inst.col});
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
            if (warp::laneid() == 0) {
                redAdd<Sem::RELAXED, Scope::GPU>(&g.Bar[g.dev_idx][{inst.layer, opcode - 1, inst.row, 0}], 1);
            }
        }
    };
};

struct up_matmul_gmem_waiter {
    template <typename Config, typename Globals, typename instruction_t>
    static __device__ inline void gmem_wait(const Globals &g, state<Config> &s, instruction_t &inst) {
        int expected_val = Globals::matmul_batch_block_size;
        wait_on_barrier<Scope::SYS>(&g.Bar[g.dev_idx][{inst.layer, OPCODE_MlpNorm - 1, inst.row, 0}],
                        expected_val, "up_matmul_gmem_waiter", "dev=%d, layer=%d, row=%d",
                        g.dev_idx, inst.layer, inst.row);
    }
};
