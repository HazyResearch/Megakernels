#include "llama.cuh"
#include "matmul_pipeline.cuh"

using namespace kittens;
using namespace megakernel;

template <auto InputActivationsPtr, auto WeightsPtr, auto OutputActivationsPtr, int iters, int _opcode,
          typename gmem_waiter, bool reduce_scatter, typename Config, typename Globals>
struct MatMulAddOp {
    static constexpr int opcode = _opcode;
    using output_tile = st_bf<64, 256>;

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

    using matmul_pipeline =
        matmul_pipeline<Config, Globals, parsed_instruction, gmem_waiter, InputActivationsPtr, WeightsPtr, iters>;

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
            warp::sync();
            warp::arrive(output_arrived(s));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            auto &OutputActivations = g.*OutputActivationsPtr;
            wait(output_arrived(s), 0);
#ifdef LLAMA_DETERMINISTIC
            if constexpr (reduce_scatter) {
                // wait barrier == g.dev_idx * constant
                int2 local_idx_info = global_block_idx_to_local_block_info(g, inst.row);
                int target_dev_idx = local_idx_info.x;
                int target_dev_local_row = local_idx_info.y;
                (void)target_dev_local_row;
                uint64_t start_time = clock64();
                int expected_val = g.dev_idx * Globals::num_output_blocks;
                while(*(volatile int *)&g.Bar[target_dev_idx][{inst.layer, OPCODE_DownProjResidual - 1, target_dev_local_row, 0}] < expected_val) {
                    __nanosleep(100);
                    uint64_t end_time = clock64();
                    if (end_time - start_time > SPIN_LOOP_TIMEOUT_CYCLES) {
                        if (laneid() == 0) {
                            printf("[DEADLOCK] matmul_adds.cu:95 LLAMA_DETERMINISTIC - dev=%d, target_dev=%d, layer=%d, row=%d, current=%d, expected=%d\n",
                                   g.dev_idx, target_dev_idx, inst.layer, target_dev_local_row,
                                   *(volatile int *)&g.Bar[target_dev_idx][{inst.layer, OPCODE_DownProjResidual - 1, target_dev_local_row, 0}],
                                   expected_val);
                        }
                        warp::sync();
                        __nanosleep(1000000); // 1ms for flush
                        warp::sync();
                        asm volatile("trap;");
                    }
                }
                fence_acquire_sys();
            }
#endif
            if (laneid() == 0) {
                for (int i = 0; i < 2; i++) {
                    auto &out_smem = matmul_pipeline::get_output_tile(s, i);
                    s.storer_record(STORE_EVENT);
                    if constexpr (reduce_scatter) {
                        int2 local_idx_info = global_block_idx_to_local_block_info(g, inst.row);
                        int target_dev_idx = local_idx_info.x;
                        int target_dev_local_row = local_idx_info.y;

                        tma::store_add_async(OutputActivations[target_dev_idx], out_smem,
                                             {2 * target_dev_local_row + i, inst.col});
                    } else {
                        tma::store_add_async(OutputActivations[g.dev_idx], out_smem,
                                             {2 * inst.local_row + i, inst.col});
                    }
                    tma::store_async_read_wait();  // to release pages faster
                    s.storer_record(READY_EVENT);
                    s.finish_page(matmul_pipeline::get_output_page(s, i), Config::NUM_CONSUMER_WARPS);
                }
            }
            tma::store_async_wait();
            gmem_waiter::inc_barrier(g, s, inst);
        }
    };
};

struct o_proj_gmem_waiter {
    template <typename Config, typename Globals, typename instruction_t>
    static __device__ inline void gmem_wait(const Globals &g, state<Config> &s, instruction_t &inst) {
        int expected_val = (Globals::matmul_batch_block_size * Globals::num_attention_heads);
        wait_on_barrier<Scope::SYS>(&g.Bar[g.dev_idx][{inst.layer, OPCODE_GQA_AttentionDecode - 1, inst.local_row, 0}],
            expected_val, "o_proj_gmem_waiter",
            "dev=%d, layer=%d, row=%d",
            g.dev_idx, inst.layer, inst.local_row);
    }

    template <typename Config, typename Globals, typename instruction_t>
    static __device__ inline void inc_barrier(const Globals &g, state<Config> &s, instruction_t &inst) {
        if (laneid() != 0) {
            return;
        }

        fence<Sem::RELEASE, Scope::GPU>();

        s.storer_record(STORE2_EVENT);
        redAdd<Sem::RELAXED, Scope::GPU>(&g.Bar[g.dev_idx][{inst.layer, OPCODE_O_ProjResidual - 1, inst.local_row, 0}], 1);
    }
};

template <typename Config, typename Globals>
struct o_proj : MatMulAddOp<&Globals::attn_out, &Globals::o_weights, &Globals::hidden_states,
                            Globals::hidden_dim / PIPELINE_K_DIM, OPCODE_O_ProjResidual, o_proj_gmem_waiter, false,
                            Config, Globals> {};

struct downproj_gmem_waiter {
    template <typename Config, typename Globals, typename instruction_t>
    static __device__ inline void gmem_wait(const Globals &g, state<Config> &s, instruction_t &inst) {
        int expected_val = (Globals::intermediate_dim / Globals::matmul_out_block_size / Globals::num_devices);
        wait_on_barrier<Scope::GPU>(&g.Bar[g.dev_idx][{inst.layer, OPCODE_UpMatmul - 1, inst.row, 0}],
            expected_val, "downproj_gmem_waiter",
            "dev=%d, layer=%d, row=%d",
            g.dev_idx, inst.layer, inst.row);
    }

    template <typename Config, typename Globals, typename instruction_t>
    static __device__ inline void inc_barrier(const Globals &g, state<Config> &s, instruction_t &inst) {
        if (laneid() != 0) {
            return;
        }

        fence<Sem::RELEASE, Scope::SYS>();

        int2 local_idx_info = global_block_idx_to_local_block_info(g, inst.row);
        int target_dev_idx = local_idx_info.x;
        int target_row = local_idx_info.y;
        (void)target_row;

        s.storer_record(STORE2_EVENT);
        redAdd<Sem::RELAXED, Scope::SYS>(&g.Bar[target_dev_idx][{inst.layer, OPCODE_DownProjResidual - 1, target_row, 0}], 1);
    }
};

template <typename Config, typename Globals>
struct downproj : MatMulAddOp<&Globals::silu_out, &Globals::down_weights, &Globals::hidden_states,
                              Globals::intermediate_dim / Globals::num_devices / PIPELINE_K_DIM,
                              OPCODE_DownProjResidual, downproj_gmem_waiter, true, Config, Globals> {};
