#include "llama.cuh"
#include "matmul_pipeline.cuh"

using namespace kittens;
using namespace megakernel;

struct qkv_gmem_waiter;

template <typename Config, typename Globals>
struct qkv_rope_append {
    static constexpr int opcode = OPCODE_QKV_RopeAppend;
    static constexpr int num_generated_heads = Globals::matmul_out_block_size / Globals::head_dim;                  // 2
    static constexpr int KV_COL_START = Globals::num_attention_heads / num_generated_heads / Globals::num_devices;  // 4
    static constexpr int NUM_ITERS = Globals::hidden_dim / PIPELINE_K_DIM;
    static_assert(KV_COL_START % 2 == 0, "Fix");

    static_assert(Globals::head_dim == 128, "Head dim must be 128.");

    using sv_fl_head_dim = sv_fl<Globals::head_dim>;
    using sv_bf_head_dim = sv_bf<Globals::head_dim>;
    using rv_fl_head_dim = rv_fl<Globals::head_dim>;

    static constexpr int tokens_per_block = Globals::matmul_batch_block_size;
    using int_vec = kittens::sv<int, tokens_per_block>;

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

        __device__ inline bool is_kv() const {
            // we're assuming here that there's 8 KV heads total and 8 gpus,
            // so 1KV head per gpu. so either this instruction is prepping 2
            // Q heads, or 1 K head and 1 V head (if the last gpu). in the
            // former case, we rope both Q heads. in the latter case, we
            // rope just the K head.
            return local_col >= KV_COL_START;
        }
    };

    using matmul_pipeline = matmul_pipeline<Config, Globals, parsed_instruction, qkv_gmem_waiter,
                                            &Globals::rms_rope_intermediates, &Globals::qkv_weights, NUM_ITERS>;

    __device__ static inline semaphore &rope_arrived(state<Config> &s) {
        return s.semaphores()[matmul_pipeline::SEM_COUNT];
    }

    __device__ static inline semaphore &outputs_arrived(state<Config> &s) {
        return s.semaphores()[matmul_pipeline::SEM_COUNT + 1];
    }

    __device__ static inline sv_fl_head_dim *get_rope_cos_vecs(state<Config> &s, int wg) {
        return reinterpret_cast<sv_fl_head_dim *>(&matmul_pipeline::get_output_tile(s, wg));
    }
    __device__ static inline sv_fl_head_dim *get_rope_sin_vecs(state<Config> &s, int wg) {
        return reinterpret_cast<sv_fl_head_dim *>(&matmul_pipeline::get_output_tile(s, 2 + wg));
    }

    __device__ static inline int_vec &get_position_ids(state<Config> &s) {
        return *reinterpret_cast<int_vec *>(s.scratch());
    }

    __device__ static inline int_vec &get_append_indices(state<Config> &s) {
        return *reinterpret_cast<int_vec *>((uint8_t *)(s.scratch()) + sizeof(int_vec));
    }

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query) {
            return matmul_pipeline::get_output_lid(query);  // there's actually another layer of indirection here.
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            if(laneid() == 0) init_semaphore(rope_arrived(s), WARP_THREADS);
            if(laneid() == 0) init_semaphore(outputs_arrived(s), Config::NUM_CONSUMER_WARPS);
            return matmul_pipeline::init_semaphores(s) + 2;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};

            semaphore &rope_arrived_sem = rope_arrived(s);
            tma::expect_bytes(rope_arrived_sem, sizeof(float) * 128 * 256 / WARP_THREADS);

            matmul_pipeline::template loader_loop(s, g, inst.layer);
            s.loader_record(LOAD2_EVENT);

            warp::sync();

            int_vec &position_ids_scratch = get_position_ids(s);
            int_vec &append_indices_scratch = get_append_indices(s);

            warp::load(position_ids_scratch, g.position_ids, {inst.row});
            warp::load(append_indices_scratch, g.kv_append_indices, {inst.row});

            load_async_wait();
            group<9>::sync(2);
            group<9>::sync(1); // the sequential 2, then 1 seems weird but is technically needed for formal correctness.

//             // gives us 1 free page (finishes the N - 4th iteration) - see the
//             // matmul loop for the release sequence.
//             matmul_pipeline::loader_input_wait(s, NUM_ITERS);
//             // gives us 2 more pages
//             matmul_pipeline::loader_input_wait(s, NUM_ITERS + 1);
//             // gives us 1 more page
//             matmul_pipeline::loader_input_wait(s, NUM_ITERS + 2);
//             // gives us final page
//             matmul_pipeline::loader_input_wait(s, NUM_ITERS + 3);

//             s.loader_record(LOAD2_EVENT);
// #pragma unroll
//             for (int i = laneid(); i < 64; i += WARP_THREADS) {
//                 int position_id = position_ids_scratch[i];
//                 tma::load_async(get_rope_cos_vecs(s, 0)[i], g.rope_cos, {position_id, 0}, rope_arrived_sem);
//                 if(position_id != i%8) asm volatile("trap;"); // TODO: REMOVE
//             }

// #pragma unroll
//             for (int i = laneid(); i < 64; i += WARP_THREADS) {
//                 int position_id = position_ids_scratch[64 + i];
//                 tma::load_async(get_rope_cos_vecs(s, 1)[i], g.rope_cos, {position_id, 0}, rope_arrived_sem);
//                 if(position_id != i%8) asm volatile("trap;"); // TODO: REMOVE
//             }

// #pragma unroll
//             for (int i = laneid(); i < 64; i += WARP_THREADS) {
//                 int position_id = position_ids_scratch[i];
//                 tma::load_async(get_rope_sin_vecs(s, 0)[i], g.rope_sin, {position_id, 0}, rope_arrived_sem);
//                 if(position_id != i%8) asm volatile("trap;"); // TODO: REMOVE
//             }

// #pragma unroll
//             for (int i = laneid(); i < 64; i += WARP_THREADS) {
//                 int position_id = position_ids_scratch[64 + i];
//                 tma::load_async(get_rope_sin_vecs(s, 1)[i], g.rope_sin, {position_id, 0}, rope_arrived_sem);
//                 if(position_id != i%8) asm volatile("trap;"); // TODO: REMOVE
//             }
//             warp::sync();
            s.loader_record(READY_EVENT);

            warp::arrive(s.instruction_fetch_ready,
                         Config::NUM_CONSUMER_WARPS);  // this actually doesn't need to be perfectly pipelined since the
                                                       // epilogue is long.
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {}
    };

    using constorer = group<Config::NUM_CONSUMER_WARPS + 1>;

    __device__ static inline void apply_rope_inplace(rv_fl_head_dim &input, const rv_fl_head_dim &rope_cos,
                                                     const rv_fl_head_dim &rope_sin) {
        constexpr int head_dim = Globals::head_dim;
        static_assert(head_dim % 32 == 0, "Head dim must be divisible by 64.");
        constexpr int num_per_thread = head_dim / 32;

        rv_fl_head_dim rotated;

        // rotate - need to access the internal data array
        warp::sync();
#pragma unroll
        for (int i = 0; i < num_per_thread; i++) {
            rotated[{i, 0}] = __shfl_xor_sync(0xffffffff, input[{i, 0}], 1);
            if (laneid() % 2 == 0) rotated[{i, 0}] *= -1.f;
        }
        warp::sync();

        // rope = vals * cos + rotated_vals * sin
        warp::mul(input, input, rope_cos);
        warp::mul(rotated, rotated, rope_sin);
        warp::add(input, input, rotated);
    }

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            static_assert(Globals::num_devices == 8, "Fix this function.");
            static_assert(Config::NUM_CONSUMER_WARPS == 8, "Fix this function.");

            parsed_instruction inst{s};

            using matmul_rt = rt_fl<16, 256>;
            using matmul_st = st_fl<16, 256>;

            rv_fl<128> out_vecs[16][2];

            matmul_rt out_fl = matmul_pipeline::matmul_loop<6>(s, g);

            sv_fl_head_dim *rope_cos_vecs_smem =
                get_rope_cos_vecs(s, warpgroup::groupid());  // Note this pointer is different for
                                                             // different warpgroups
            sv_fl_head_dim *rope_sin_vecs_smem =
                get_rope_sin_vecs(s, warpgroup::groupid());  // Note this pointer is different for
                                                             // different warpgroups

            group<9>::sync(2);
            group<9>::sync(1); // the sequential 2, then 1 seems weird but is technically needed for formal correctness.

            #pragma unroll
            for(int i = 0; i < 16; i++) {
                int position_id = get_position_ids(s)[warpid()*16 + i];
                warp::load(rope_cos_vecs_smem[warpgroup::warpid()*16 + i], g.rope_cos, {position_id, 0});
                warp::load(rope_sin_vecs_smem[warpgroup::warpid()*16 + i], g.rope_sin, {position_id, 0});
            }

            matmul_st &convert_tile = (reinterpret_cast<matmul_st *>(
                &s.pages[matmul_pipeline::get_output_page(s, 4 + (warpgroup::warpid() / 2))]))[warpgroup::warpid() % 2];

            group<8>::sync(0);
            if (warpgroup::groupid() == 0) {
                warp::store(convert_tile, out_fl);
                warp::sync();
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    warp::load(out_vecs[i][0], convert_tile, {i, 0});
                    warp::load(out_vecs[i][1], convert_tile, {i, 128});
                }
                group<8>::sync(0);
            } else {
                group<8>::sync(0);
                warp::store(convert_tile, out_fl);
                warp::sync();
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    warp::load(out_vecs[i][0], convert_tile, {i, 0});
                    warp::load(out_vecs[i][1], convert_tile, {i, 128});
                }
            }

            // store the result on top of cos/sin
            // (now that it's in bf16, we can fit in two pages).
            sv_bf_head_dim *output_vecs = reinterpret_cast<sv_bf_head_dim *>(
                &s.pages[matmul_pipeline::get_output_page(s, 4 + warpgroup::groupid())]);

            rv_fl_head_dim rope_cos_vec, rope_sin_vec, activations_vec;

            // wait(rope_arrived(s), 0);
            load_async_wait();
            group<8>::sync(0);

#pragma unroll
            for (int i = 0; i < 16; i++) {
                int token_idx = warpgroup::warpid() * 16 + i;

                warp::load(rope_cos_vec, rope_cos_vecs_smem[token_idx]);
                warp::load(rope_sin_vec, rope_sin_vecs_smem[token_idx]);

                apply_rope_inplace(out_vecs[i][0], rope_cos_vec, rope_sin_vec); // there is a warp sync within this
                if (!inst.is_kv()) {
                    apply_rope_inplace(out_vecs[i][1], rope_cos_vec, rope_sin_vec);
                }
            }
#pragma unroll
            for (int i = 0; i < 16; i++) {
                int token_idx = warpgroup::warpid() * 16 + i;
                warp::store(output_vecs[2 * token_idx + 0], out_vecs[i][0]);
                warp::store(output_vecs[2 * token_idx + 1], out_vecs[i][1]);
            }

            group<8>::sync(1);

            warp::arrive(outputs_arrived(s));

            group<8>::sync(1);

            // release the cos & sin pages
            s.warp_finish_page(matmul_pipeline::get_output_page(s, 0 + warpgroup::groupid()), 2);
            s.warp_finish_page(matmul_pipeline::get_output_page(s, 2 + warpgroup::groupid()), 2);
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            wait(outputs_arrived(s), 0);

            static_assert(Globals::num_devices == 8, "Fix this function.");
            parsed_instruction inst{s};

            sv_bf_head_dim *output_vecs[2] = {
                reinterpret_cast<sv_bf_head_dim *>(&s.pages[matmul_pipeline::get_output_page(s, 4)]),
                reinterpret_cast<sv_bf_head_dim *>(&s.pages[matmul_pipeline::get_output_page(s, 5)])};

            bool is_kv = inst.is_kv();

            auto &append_indices = get_append_indices(s);

            s.storer_record(STORE_EVENT);
#pragma unroll
            for (int j = 0; j < 2; j++) {
                auto &vecs = output_vecs[j];
#pragma unroll
                for (int i = laneid(); i < 64; i += WARP_THREADS) {
                    int idx = j * 64 + i;
                    auto &first_vec = vecs[2 * i];
                    auto &second_vec = vecs[2 * i + 1];

                    if (!is_kv) {
                        tma::store_async(g.q_post_rope, first_vec,
                                         {inst.row * Globals::matmul_batch_block_size + idx, 2 * inst.local_col});
                        tma::store_async(g.q_post_rope, second_vec,
                                         {inst.row * Globals::matmul_batch_block_size + idx, 2 * inst.local_col + 1});
                    } else {
                        // first vec is K, second vec is V

                        // token idx
                        auto append_idx = append_indices[idx];
                        auto page_idx = append_idx / Globals::kv_page_size;
                        auto offset_in_page = append_idx % Globals::kv_page_size;

                        auto kv_head_idx_on_this_gpu = 0;

                        tma::store_async(
                            g.k_cache, first_vec,
                            {g.num_pages * inst.layer + page_idx, offset_in_page, kv_head_idx_on_this_gpu, 0});

                        tma::store_async(
                            g.v_cache, second_vec,
                            {g.num_pages * inst.layer + page_idx, offset_in_page, kv_head_idx_on_this_gpu, 0});
                    }
                }
            }

            tma::store_async_read_wait();
            warp::sync();
            s.storer_record(READY_EVENT);
            // release pages we've been using
            s.warp_finish_page(matmul_pipeline::get_output_page(s, 4), Config::NUM_CONSUMER_WARPS);
            s.warp_finish_page(matmul_pipeline::get_output_page(s, 5), Config::NUM_CONSUMER_WARPS);

            tma::store_async_wait();
            fence<Sem::RELEASE, Scope::GPU>();

            int start_bar = (inst.local_col * Globals::matmul_out_block_size) / Globals::head_dim;
            s.storer_record(STORE2_EVENT);

            redAdd<Sem::RELAXED, Scope::GPU>(&g.Bar[g.dev_idx][{inst.layer, opcode - 1, inst.local_row, is_kv}], 1);
        }
    };
};

struct qkv_gmem_waiter {
    template <typename Config, typename Globals, typename instruction_t>
    static __device__ inline void gmem_wait(const Globals &g, state<Config> &s, instruction_t &inst) {
        int expected_val = Globals::matmul_batch_block_size;
        wait_on_barrier<Scope::SYS>(&g.Bar[g.dev_idx][{inst.layer, OPCODE_AttnNorm - 1, inst.row, 0}],
                        expected_val, "qkv_gmem_waiter", "dev=%d, layer=%d, row=%d",
                        g.dev_idx, inst.layer, inst.row);
    }
};
