#include "llama.cuh"

using namespace kittens;
using namespace megakernel;

// Masked load for Q
template <int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD = coord<RT>>
__device__ inline static void masked_q_warp_load(RT &dst, const GL &src, const COORD &idx, int num_kv_heads) {
    using T2 = RT::dtype;
    using U = typename GL::dtype;

#ifdef KITTENS_HOPPER
    static_assert(!std::is_same_v<T2, fp8e4m3_4> && !std::is_same_v<T2, fp8e5m2_4>, "Unsupported type for load/store");
#endif

    U *src_ptr = (U *)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    using U2 = base_types::packing<U>::packed_type;
    int warp_laneid = threadIdx.x % WARP_THREADS;
    int kv_offset = warp_laneid / 4;
    if (kv_offset < num_kv_heads) {
#pragma unroll
        for (int j = 0; j < dst.width; j++) {
            int col = j * dst.tile_size_col + 2 * (warp_laneid % 4);
            dst.tiles[0][j].data[0] =
                base_types::convertor<T2, U2>::convert(*(U2 *)(&src_ptr[kv_offset * 128 + col + 0]));
            dst.tiles[0][j].data[2] =
                base_types::convertor<T2, U2>::convert(*(U2 *)(&src_ptr[kv_offset * 128 + col + 8]));
        }
    }
}

template <typename Config, typename Globals>
struct attention_decode {
    static constexpr int opcode = OPCODE_GQA_AttentionDecode;
    static constexpr int NUM_STAGES = 3;
    static constexpr int GQA_RATIO = Globals::num_attention_heads / Globals::num_kv_heads;
    static constexpr int NUM_ATTN_HEADS_PER_DEVICE = Globals::num_attention_heads / Globals::num_devices;
    static_assert(NUM_ATTN_HEADS_PER_DEVICE == 8, "Fix");

    static_assert(GQA_RATIO == 8, "GQA_RATIO must be 8.");
    static_assert(NUM_STAGES <= Config::NUM_PAGES, "Not enough pages. Time to actually use full pages.");

    static constexpr int head_dim = Globals::head_dim;
    static constexpr int kv_block_size = 16;
    static constexpr int kv_page_size = Globals::kv_page_size;
    static constexpr int iters_per_page = kv_page_size / kv_block_size;

    static constexpr int MAX_SEQS_PER_INSTRUCTION = 8;
    static constexpr int SEM_COUNT_PER_SEQ = 1 + 4 * NUM_STAGES;
    static constexpr int MAX_SEM_COUNT_PER_INSTRUCTION = MAX_SEQS_PER_INSTRUCTION * SEM_COUNT_PER_SEQ;

    static_assert(MAX_SEM_COUNT_PER_INSTRUCTION <= Config::DYNAMIC_SEMAPHORES, "Not enough semaphores.");

    using q_rt = rt_bf<16, head_dim>;  // only `GQA_RATIO` rows are used
    using q_st = st_bf<16, head_dim>;  // only `GQA_RATIO` rows are used
    using k_rt = rt_bf<16, head_dim>;
    using v_rt = rt_bf<16, head_dim, col_l>;
    using kv_st = st_bf<kv_block_size, head_dim>;
    using attn_fl_rt = rt_fl<16, 16>;                  // only `GQA_RATIO` values are used
    using attn_bf_rt = rt_bf<16, 16>;                  // only `GQA_RATIO` values are used
    using max_vec_rv = col_vec<rt_fl<16, head_dim>>;   // only `GQA_RATIO` values are used
    using max_vec_sv = sv_fl<16>;                      // only `GQA_RATIO` values are used
    using norm_vec_rv = col_vec<rt_fl<16, head_dim>>;  // only `GQA_RATIO` values are used
    using norm_vec_sv = sv_fl<16>;                     // only `GQA_RATIO` values are used
    using l_rv = col_vec<rt_fl<16, head_dim>>;         // only `GQA_RATIO` values are used
    using l_sv = sv_fl<16>;                            // only `GQA_RATIO` values are used
    using o_rt = rt_fl<16, head_dim>;                  // only `GQA_RATIO` rows are used
    using o_rt_bf = rt_bf<16, head_dim>;               // only `GQA_RATIO` rows are used
    using o_sv = sv_bf<head_dim>;

    __device__ static inline int get_layer_idx(state<Config> &s) { return s.instruction()[1]; }
    __device__ static inline int get_num_seqs_in_instruction(state<Config> &s) { return s.instruction()[2] / 2; }
    __device__ static inline int get_global_seq_idx(state<Config> &s, int seq_id) {
        return s.instruction()[3 + 2 * seq_id];
    }
    __device__ static inline int get_kv_head_idx(state<Config> &s, int seq_id) {
        return s.instruction()[3 + 2 * seq_id + 1];
    }
    __device__ static inline int get_batch_idx(state<Config> &s, const Globals &g, int seq_id) {
        auto global_seq_idx = get_global_seq_idx(s, seq_id);
        return global_seq_idx + g.num_prefill_tokens;
    }

    __device__ static inline semaphore &O_arrived(state<Config> &s, int seq_id) {
        return s.semaphores()[seq_id * SEM_COUNT_PER_SEQ];
    }
    __device__ static inline semaphore &K_arrived(state<Config> &s, int seq_id, int stage) {
        return s.semaphores()[seq_id * SEM_COUNT_PER_SEQ + 1 + stage * 2];
    }
    __device__ static inline semaphore &V_arrived(state<Config> &s, int seq_id, int stage) {
        return s.semaphores()[seq_id * SEM_COUNT_PER_SEQ + 1 + stage * 2 + 1];
    }
    __device__ static inline semaphore &K_finished(state<Config> &s, int seq_id, int stage) {
        return s.semaphores()[seq_id * SEM_COUNT_PER_SEQ + 1 + NUM_STAGES * 2 + stage * 2];
    }
    __device__ static inline semaphore &V_finished(state<Config> &s, int seq_id, int stage) {
        return s.semaphores()[seq_id * SEM_COUNT_PER_SEQ + 1 + NUM_STAGES * 2 + stage * 2 + 1];
    }

    __device__ static inline int get_K_page(state<Config> &s, int stage) { return s.pid(0 + stage); }
    __device__ static inline int get_V_page(state<Config> &s, int stage) { return s.pid(3 + stage); }
    __device__ static inline kv_st &as_kv_st(state<Config> &s, int seq_id, int page) {
        return reinterpret_cast<kv_st *>(s.pages[page].data)[seq_id];
    }

    template <bool add, ducks::sv::all SV, ducks::rt::all RT>
    __device__ static inline void store_8_rows(SV(*dst), const RT &src) {
        static_assert(RT::rows == 16, "src rows must be 16.");
        static_assert(SV::length == src.cols, "dst length must match src cols.");

        using T = RT::T;
        using T2 = RT::T2;
        using U = SV::T;
        using U2 = SV::T2;

        uint32_t dst_ptr[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) dst_ptr[i] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[i].data[0]));

        int laneid = kittens::laneid();
        int local_row_idx = laneid / 4;
        int local_col_idx = laneid % 4;

        for (int j = 0; j < src.width; j++) {
            U2 tmp[2];
            tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[0]);
            tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[2]);
            int col_idx = local_col_idx * 2 + j * 16;
            if constexpr (add) {
                atomicAdd(&dst[local_row_idx][0] + (col_idx + 0), tmp[0].x);
                atomicAdd(&dst[local_row_idx][0] + (col_idx + 1), tmp[0].y);
                atomicAdd(&dst[local_row_idx][0] + (col_idx + 8), tmp[1].x);
                atomicAdd(&dst[local_row_idx][0] + (col_idx + 9), tmp[1].y);
            } else {
                // *((U2*)&dst[local_row_idx][0] + col_idx/2 + 0) = tmp[0];
                // *((U2*)&dst[local_row_idx][0] + col_idx/2 + 4) = tmp[1];
                move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx + 8), tmp[1]);
            }
        }
    }

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query) {
            // int ret_order[Config::NUM_PAGES] = {0,1,2,3,4,5};
            // return ret_order[query];
            return query;
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            int num_seqs = get_num_seqs_in_instruction(s);
            if(laneid() < num_seqs) {
                init_semaphore(O_arrived(s, laneid()), 1);
            }
            if(laneid() < NUM_STAGES * num_seqs) {
                int seq_id = laneid() / NUM_STAGES;
                int stage = laneid() % NUM_STAGES;
                init_semaphore(K_arrived(s, seq_id, stage), 1);
                init_semaphore(V_arrived(s, seq_id, stage), 1);
                init_semaphore(K_finished(s, seq_id, stage), 1);
                init_semaphore(V_finished(s, seq_id, stage), 1);
            }
            return num_seqs * SEM_COUNT_PER_SEQ;
        }
    };
    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            int laneid = warp::laneid();

            int num_seqs = get_num_seqs_in_instruction(s);

            warp::arrive(s.instruction_fetch_ready, Config::NUM_CONSUMER_WARPS);
            if (laneid < num_seqs) {
                int seq_id = laneid;
                int global_seq_idx = get_global_seq_idx(s, seq_id);
                int batch_idx = get_batch_idx(s, g, seq_id);

                int indptr_start = g.decode_kv_indptr[{global_seq_idx}];
                int indptr_end = g.decode_kv_indptr[{global_seq_idx + 1}];
                int num_pages = indptr_end - indptr_start;
                int total_attn_blocks = (
                    ((num_pages - 1) * iters_per_page) + 
                    (g.decode_kv_last_page_len[{global_seq_idx}] + kv_block_size - 1) / kv_block_size
                );

                int layer_idx = get_layer_idx(s);
                int batch_block_idx = batch_idx / Globals::matmul_batch_block_size;
                int kv_head_idx = get_kv_head_idx(s, seq_id);

                // Run the pipeline!
                for (int i = 0; i < total_attn_blocks; ++i) {
                    int stage = i % NUM_STAGES;
                    int k_page = get_K_page(s, stage);
                    int v_page = get_V_page(s, stage);
                    kv_st &K_smem = as_kv_st(s, seq_id, k_page);
                    kv_st &V_smem = as_kv_st(s, seq_id, v_page);

                    if (i >= NUM_STAGES) {
                        wait(K_finished(s, seq_id, stage), (i / NUM_STAGES - 1) % 2);
                        wait(V_finished(s, seq_id, stage), (i / NUM_STAGES - 1) % 2);
                    } else {
                        s.wait_page_ready(k_page);
                        s.wait_page_ready(v_page);
                    }

                    int decode_kv_indices_slot = indptr_start + (i / iters_per_page);
                    (void)decode_kv_indices_slot;

                    int iter_in_page = i % iters_per_page;
                    int kv_page_index = g.decode_kv_indices[{decode_kv_indices_slot}];

                    tma::expect(K_arrived(s, seq_id, stage), K_smem);
                    if (i == 0) {
                        s.loader_record(WAIT_EVENT);
                        wait_on_barrier<Scope::GPU>(&g.Bar[g.dev_idx][{layer_idx, OPCODE_QKV_RopeAppend - 1, batch_block_idx, 1}],
                                        32, "loader KV barrier", "dev=%d, layer=%d, batch_block=%d",
                                        g.dev_idx, layer_idx, batch_block_idx);
                        s.loader_record(READY_EVENT);
                    }
                    s.loader_record(LOAD_EVENT);

                    tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(
                        K_smem, g.k_cache, {(int)g.num_pages * layer_idx + kv_page_index, iter_in_page, kv_head_idx, 0},
                        K_arrived(s, seq_id, stage));

                    tma::expect(V_arrived(s, seq_id, stage), V_smem);
                    tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(
                        V_smem, g.v_cache, {(int)g.num_pages * layer_idx + kv_page_index, iter_in_page, kv_head_idx, 0},
                        V_arrived(s, seq_id, stage));
                }
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
        static __device__ __forceinline__ void finish_with_pages(state<Config> &s) {
            for (int i = 0; i < NUM_STAGES; i++) {
                int k_page = get_K_page(s, i), v_page = get_V_page(s, i);
                s.wait_page_ready(k_page);
                warp::sync();
                s.warp_finish_page(k_page, 1);
                if (i != NUM_STAGES - 1) {
                    s.wait_page_ready(v_page);
                    warp::sync();
                    s.warp_finish_page(v_page,
                                       1);  // Leave the last V page for stores.
                }
            }
        }

        static __device__ void run(const Globals &g, state<Config> &s) {
            static_assert(Config::NUM_CONSUMER_WARPS == 8, "Fix this function.");

            int seq_id = warpid();
            int num_seqs = get_num_seqs_in_instruction(s);

            bool is_used = seq_id < num_seqs;

            if (!is_used) {
                finish_with_pages(s);
                group<8>::sync(0);
                return;
            }

            // Setup
            q_rt Q_reg;

            int layer_idx = get_layer_idx(s);
            int global_seq_idx = get_global_seq_idx(s, seq_id);
            int batch_idx = get_batch_idx(s, g, seq_id);
            int batch_block_idx = batch_idx / Globals::matmul_batch_block_size;
            int kv_head_idx = get_kv_head_idx(s, seq_id);

            wait_on_barrier<Scope::GPU>(&g.Bar[g.dev_idx][{layer_idx, OPCODE_QKV_RopeAppend - 1, batch_block_idx, 0}],
                            128, "consumer QKV barrier", "dev=%d, layer=%d, batch_block=%d",
                            g.dev_idx, layer_idx, batch_block_idx);
            warp::sync();
            s.consumer_record(LOAD2_EVENT);
            // Initiate the load on Q
            // This is a little hacky but should work. Basically, since we only
            // need GQA_RATIO rows, we can just mask a bunch of threads.
            // load_Q_async(Q_smem, g.q_post_rope, inst.batch_idx,
            // inst.kv_head_idx * GQA_RATIO);
            warp::zero(Q_reg);
            masked_q_warp_load<dim::ROW>(Q_reg, g.q_post_rope, coord<>{batch_idx, kv_head_idx * GQA_RATIO * head_dim},
                                         GQA_RATIO);  // only load GQA_RATIO rows

            k_rt K_reg;
            v_rt V_reg;
            o_rt O_reg;
            attn_fl_rt attn_fl_reg;
            attn_bf_rt attn_bf_reg;
            max_vec_rv max_vec_reg;
            max_vec_rv scaled_max_vec_reg;
            max_vec_rv last_scaled_max_vec_reg;
            max_vec_rv diff_scaled_max_vec_reg;
            norm_vec_rv norm_vec_reg;
            warp::neg_infty(max_vec_reg);
            warp::neg_infty(last_scaled_max_vec_reg);
            warp::zero(norm_vec_reg);
            warp::zero(O_reg);

            float softmax_temp = g.attn_scale * 1.44269504089f;  // 1 / (sqrt(D_h) * ln(2))

            // Run the pipeline!
            int indptr_start = g.decode_kv_indptr[{global_seq_idx}];
            int indptr_end = g.decode_kv_indptr[{global_seq_idx + 1}];
            int num_pages = indptr_end - indptr_start;
            int num_attn_blocks = (
                ((num_pages - 1) * iters_per_page) + 
                (g.decode_kv_last_page_len[{global_seq_idx}] + kv_block_size - 1) / kv_block_size
            );

            int last_page_len = g.decode_kv_last_page_len[{global_seq_idx}];
            int sequence_length = (num_pages - 1) * kv_page_size + last_page_len;

            for (int i = 0; i < num_attn_blocks; ++i) {
                int stage = i % NUM_STAGES;
                int k_page = get_K_page(s, stage);
                int v_page = get_V_page(s, stage);
                kv_st &K_smem = as_kv_st(s, seq_id, k_page);
                kv_st &V_smem = as_kv_st(s, seq_id, v_page);

                // Perform Q @ K.T
                warp::zero(attn_fl_reg);
                warp::wait(K_arrived(s, seq_id, stage), (i / NUM_STAGES) % 2);
                s.consumer_record(COMPUTE_EVENT);
                warp::load(K_reg, K_smem);
                warp::mma_ABt(attn_fl_reg, Q_reg, K_reg, attn_fl_reg);
                warp::sync();
                warp::arrive(K_finished(s, seq_id, stage));

                // Mask out invalid positions at the end
                if (i >= num_attn_blocks - iters_per_page) {
                    int remaining_length = sequence_length - (i * kv_block_size);
                    warp::apply(attn_fl_reg, attn_fl_reg, [remaining_length] __device__(int row, int col, float val) {
                        return (col >= remaining_length) ? -999999999999.f : val;
                    });
                }

                // Obtain maximums per row (which is per head)
                warp::row_max(max_vec_reg, attn_fl_reg,
                              max_vec_reg);  // includes previous max

                // Scale attention block and maximums by sqrt(D_h)
                warp::mul(attn_fl_reg, attn_fl_reg, softmax_temp);
                warp::mul(scaled_max_vec_reg, max_vec_reg, softmax_temp);

                // Calculate softmax numerator
                warp::sub_row(attn_fl_reg, attn_fl_reg, scaled_max_vec_reg);
                warp::exp2(attn_fl_reg, attn_fl_reg);

                // Calculate softmax denominator
                warp::sub(diff_scaled_max_vec_reg, last_scaled_max_vec_reg, scaled_max_vec_reg);
                warp::exp2(diff_scaled_max_vec_reg, diff_scaled_max_vec_reg);

                // Normalize and accumulate numerator (A @ V)
                warp::mul_row(O_reg, O_reg, diff_scaled_max_vec_reg);
                warp::wait(V_arrived(s, seq_id, stage), (i / NUM_STAGES) % 2);

                warp::load(V_reg, V_smem);
                warp::sync();
                warp::arrive(V_finished(s, seq_id, stage));
                warp::copy(attn_bf_reg,
                           attn_fl_reg);  // Convert to bf16 to do matmul
                warp::mma_AB(O_reg, attn_bf_reg, V_reg, O_reg);

                // Normalize and accumulate demoniator
                warp::mul(norm_vec_reg, norm_vec_reg, diff_scaled_max_vec_reg);
                warp::row_sum(norm_vec_reg, attn_fl_reg, norm_vec_reg);

                // Save for next iteration
                warp::copy(last_scaled_max_vec_reg, scaled_max_vec_reg);
            }
            finish_with_pages(s);
            group<8>::sync(0);

            // Next normalize the output according with the current logsumexp
            warp::add(norm_vec_reg, norm_vec_reg, 1e-16f);
            warp::div_row(O_reg, O_reg, norm_vec_reg);

            s.consumer_record(STORE_EVENT);

            // Store to O_SMEM
            uint8_t(*O_smem) = reinterpret_cast<uint8_t *>(s.pages[s.pid(5)].data);  // reusing last V page

            store_8_rows<false>(
                reinterpret_cast<sv_bf<128> *>(O_smem + seq_id * Globals::head_dim * GQA_RATIO * sizeof(bf16)),
                O_reg);  // just store, no add.

            warp::sync();
            if (seq_id < num_seqs) {
                warp::arrive(O_arrived(s, seq_id));
            }
        }
    };
    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            int laneid = warp::laneid();
            int batch_size_per_dev = g.batch_size / Globals::num_devices;
            constexpr int heads_per_dev = Globals::num_attention_heads / Globals::num_devices;

            int O_page = s.pid(5);
            sv_bf<128>(*O_smem) = reinterpret_cast<sv_bf<128> *>(s.pages[O_page].data);  // reusing last V page

            int num_seqs = get_num_seqs_in_instruction(s);

            if (laneid < num_seqs) {
                int seq_id = laneid;
                int kv_head_idx = get_kv_head_idx(s, laneid);

                // auto global_seq_id = get_global_seq_idx(s, seq_id);
                // int target_dev_idx = global_seq_id / batch_size_per_dev;
                // int row = global_seq_id % batch_size_per_dev;

                int batch_idx = get_batch_idx(s, g, seq_id);
                int2 local_idx_info = global_batch_idx_to_local_idx_info(g, batch_idx);
                int target_dev_idx = local_idx_info.x;
                int row = local_idx_info.y;

                wait(O_arrived(s, seq_id), 0);
                s.storer_record(STORE_EVENT);

                for (int q_head_idx = 0; q_head_idx < GQA_RATIO; q_head_idx++) {
                    int col = g.dev_idx * heads_per_dev + kv_head_idx * GQA_RATIO + q_head_idx;
                    tma::store_async(g.attn_out[target_dev_idx], O_smem[GQA_RATIO * seq_id + q_head_idx], {row, col});
                }
            }

            tma::store_async_read_wait();  // Wait until it's read from SMEM
            warp::sync();
            s.storer_record(READY_EVENT);
            s.warp_finish_page(O_page,
                               Config::NUM_CONSUMER_WARPS);  // Finish the page.

            tma::store_async_wait();
            warp::sync();  // ensure all writes are committed
            fence<Sem::RELEASE, Scope::SYS>();
            s.storer_record(STORE2_EVENT);

            if (laneid < num_seqs) {
                int seq_id = laneid;

                int batch_idx = get_batch_idx(s, g, seq_id);
                int2 local_idx_info = global_batch_idx_to_local_idx_info(g, batch_idx);
                int target_dev_idx = local_idx_info.x;
                int local_batch_block_idx = local_idx_info.y / Globals::matmul_batch_block_size;

                redAdd<Sem::RELAXED, Scope::SYS>(&g.Bar[target_dev_idx][{get_layer_idx(s), opcode - 1, local_batch_block_idx, 0}],
                          Globals::num_attention_heads / Globals::num_kv_heads);
            }
        }
    };
};
