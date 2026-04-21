#include "llama.cuh"

using namespace kittens;
using namespace megakernel;

template <typename Config, typename Globals>
struct attention_prefill {
    static constexpr int opcode = OPCODE_GQA_AttentionPrefill;
    static constexpr int NUM_STAGES = 2;
    static constexpr int GQA_RATIO = Globals::num_attention_heads / Globals::num_kv_heads;
    static constexpr int NUM_ATTN_HEADS_PER_DEVICE = Globals::num_attention_heads / Globals::num_devices;
    static_assert(NUM_ATTN_HEADS_PER_DEVICE == 8, "Fix");

    static_assert(GQA_RATIO == 8, "GQA_RATIO must be 8.");

    static constexpr int head_dim = Globals::head_dim;
    static constexpr int kv_page_size = Globals::kv_page_size;
    using q_rt = rt_bf<16, head_dim>;
    using o_rt = rt_fl<16, head_dim>;
    using q_st = st_bf<64, head_dim>;
    using o_st = st_bf<16, head_dim>;
    using kv_st = st_bf<kv_page_size, head_dim>;
    using attn_fl_rt = rt_fl<16, kv_page_size>;
    using attn_bf_rt = rt_bf<16, kv_page_size>;
    using max_vec_rv = col_vec<rt_fl<16, head_dim>>;
    using norm_vec_rv = col_vec<rt_fl<16, head_dim>>;
    using head_vec_sv = sv_bf<128>;

    struct prefill_instruction {
        int layer_idx;
        int seq_idx;
        int prefill_block_idx;
        int token_offset;
        int kv_head_idx;
        int abs_q_row, abs_q_row_last;
        int rel_q_row, rel_q_row_last;
        int kv_indptr_start;
        int attn_blocks;
        int prefill_token_offset;
        int sequence_length;
        int q_start_idx, q_end_idx;
        int q_size;
        __device__ __inline__ prefill_instruction(const Globals &g, state<Config> &s) {
            layer_idx = s.instruction()[1];
            seq_idx = s.instruction()[2];
            prefill_block_idx = s.instruction()[3];
            prefill_token_offset = s.instruction()[4];
            kv_head_idx = s.instruction()[5];  // 0

            q_start_idx = g.prefill_qo_indptr[{seq_idx}];
            q_end_idx = g.prefill_qo_indptr[{seq_idx + 1}];

            q_size = q_end_idx - q_start_idx;

            rel_q_row = 16 * prefill_block_idx;
            rel_q_row_last = min(rel_q_row + 15, q_size - 1);

            abs_q_row = rel_q_row + q_start_idx;
            abs_q_row_last = rel_q_row_last + q_start_idx;

            kv_indptr_start = g.prefill_kv_indptr[{seq_idx}];
            sequence_length = prefill_token_offset + rel_q_row_last+1;
            attn_blocks = (sequence_length + kv_page_size - 1) / kv_page_size;
        }
    };

    __device__ static inline semaphore &Q_arrived(state<Config> &s) { return s.semaphores()[0]; }
    __device__ static inline semaphore &O_arrived(state<Config> &s) { return s.semaphores()[1]; }
    __device__ static inline semaphore &K_arrived(state<Config> &s, int stage) { return s.semaphores()[2 + stage * 2]; }
    __device__ static inline semaphore &V_arrived(state<Config> &s, int stage) {
        return s.semaphores()[2 + stage * 2 + 1];
    }
    __device__ static inline semaphore &K_finished(state<Config> &s, int stage) {
        return s.semaphores()[2 + NUM_STAGES * 2 + stage * 2];
    }
    __device__ static inline semaphore &V_finished(state<Config> &s, int stage) {
        return s.semaphores()[2 + NUM_STAGES * 2 + stage * 2 + 1];
    }

    __device__ static inline int Q_page(state<Config> &s) { return s.pid(0); }
    __device__ static inline int K_page(state<Config> &s, int stage) { return s.pid(1 + 2 * stage); }
    __device__ static inline int V_page(state<Config> &s, int stage) { return s.pid(2 + 2 * stage); }
    __device__ static inline int O_page(state<Config> &s) { return s.pid(5); }

    __device__ static inline q_st &Q(state<Config> &s, int wgid) {
        return reinterpret_cast<q_st *>(&s.pages[Q_page(s)])[wgid];
    }
    __device__ static inline kv_st &K(state<Config> &s, int stage) {
        return *(reinterpret_cast<kv_st *>(&s.pages[K_page(s, stage)]));
    }
    __device__ static inline kv_st &V(state<Config> &s, int stage) {
        return *(reinterpret_cast<kv_st *>(&s.pages[V_page(s, stage)]));
    }
    __device__ static inline o_st &O(state<Config> &s, int warpid) {
        return reinterpret_cast<o_st *>(&s.pages[O_page(s)])[warpid];
    }
    __device__ static inline head_vec_sv (&O_vecs(state<Config> &s))[8][16] {
        return reinterpret_cast<head_vec_sv(&)[8][16]>(s.pages[O_page(s)]);
    }

    __device__ static inline kv_st &as_kv_st(state<Config> &s, int seq_id, int page) {
        return reinterpret_cast<kv_st *>(s.pages[page].data)[seq_id];
    }

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query) {
            constexpr int ret_order[Config::NUM_PAGES] = {0, 1, 3, 2, 4, 5};
            return ret_order[query];
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            if(laneid() == 0) init_semaphore(Q_arrived(s), 1);
            if(laneid() == 0) init_semaphore(O_arrived(s), Config::NUM_CONSUMER_WARPS);
            if(laneid() < NUM_STAGES) {
                init_semaphore(K_arrived(s, laneid()), 1);
                init_semaphore(V_arrived(s, laneid()), 1);
                init_semaphore(K_finished(s, laneid()), Config::NUM_CONSUMER_WARPS);
                init_semaphore(V_finished(s, laneid()), Config::NUM_CONSUMER_WARPS);
            }
            return 24 + 4 * NUM_STAGES;
        }
    };
    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            int laneid = warp::laneid();
            prefill_instruction prefill_info(g, s);
            warp::arrive(s.instruction_fetch_ready, Config::NUM_CONSUMER_WARPS);

            // await and load Q's
            warp::tma::expect(Q_arrived(s), Q(s, 0), Q(s, 1));
            warp::sync();
            st_bf<16, 64>(&Q_smem_reshaped)[16] = reinterpret_cast<st_bf<16, 64>(&)[16]>(Q(s, 0));
            if (laneid < 16) {
                int col_chunk = (laneid % 8) / 4;
                int wgmma_chunk = laneid / 8;
                // This awkward indexing reconstructs two 64x128 tiles with 128B swizzling.
                int batch_block_idx = prefill_info.abs_q_row / Globals::matmul_batch_block_size;
                int batch_block_idx_last =
                    prefill_info.abs_q_row_last / Globals::matmul_batch_block_size;  // no alignment guarantees, alas
                wait_on_barrier<Scope::GPU>(&g.Bar[g.dev_idx][{prefill_info.layer_idx, OPCODE_QKV_RopeAppend - 1, batch_block_idx, 0}],
                                128, "loader QKV barrier", "dev=%d, layer=%d, batch_block=%d",
                                g.dev_idx, prefill_info.layer_idx, batch_block_idx);
                if (batch_block_idx_last != batch_block_idx) {
                    wait_on_barrier<Scope::GPU>(&g.Bar[g.dev_idx][{prefill_info.layer_idx, OPCODE_QKV_RopeAppend - 1, batch_block_idx_last, 0}],
                                    128, "loader QKV barrier last", "dev=%d, layer=%d, batch_block=%d",
                                    g.dev_idx, prefill_info.layer_idx, batch_block_idx_last);
                }
                s.wait_page_ready(Q_page(s));
                warp::sync<0xFFFF>();
                tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    Q_smem_reshaped[laneid], g.q_post_rope,
                    coord<>{prefill_info.abs_q_row, wgmma_chunk * 512 + 128 * (laneid % 4) + 64 * col_chunk}, Q_arrived(s));
            }

            auto stage_idx_for_barriers = prefill_info.prefill_token_offset / kv_page_size;

            // Run the pipeline!
            for (int i = 0; i < prefill_info.attn_blocks; ++i) {
                // At the start of the last attn block, signal the controller to make fetch happen.
                // if(i+1 == attn_blocks) warp::arrive(s.instruction_fetch_ready, Config::NUM_CONSUMER_WARPS);

                // pay gmem latency before waiting on barriers
                int kv_page_index = g.prefill_kv_indices[{prefill_info.kv_indptr_start + i}];

                int stage = i % NUM_STAGES;
                if (i >= NUM_STAGES) {
                    wait(K_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                    wait(V_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                } else {
                    s.wait_page_ready(K_page(s, stage));
                    s.wait_page_ready(V_page(s, stage));
                }

                warp::tma::expect(K_arrived(s, stage), K(s, stage));

                // if (i * kv_page_size + kv_page_size - 1 >= prefill_info.prefill_token_offset &&
                //     (i - 1) * kv_page_size + kv_page_size - 1 < prefill_info.prefill_token_offset) {

                if (i == stage_idx_for_barriers) {
                    // this is the point where we need to wait for current KV cache to be generated.
                    s.loader_record(WAIT_EVENT);
                    int q_start_idx = g.prefill_qo_indptr[{prefill_info.seq_idx}];
                    int q_end_idx = g.prefill_qo_indptr[{prefill_info.seq_idx + 1}];
                    for (int j = q_start_idx + laneid * Globals::matmul_batch_block_size; j < q_end_idx;
                         j += WARP_THREADS * Globals::matmul_batch_block_size) {
                        int check_row = j / Globals::matmul_batch_block_size;
                        wait_on_barrier<Scope::GPU>(&g.Bar[g.dev_idx][{prefill_info.layer_idx, OPCODE_QKV_RopeAppend - 1, check_row, 1}],
                                        32, "loader KV barrier", "dev=%d, layer=%d, check_row=%d",
                                        g.dev_idx, prefill_info.layer_idx, check_row);
                    }
                    warp::sync();
                    s.loader_record(READY_EVENT);
                }
                s.loader_record(LOAD_EVENT);
                warp::tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(
                    K(s, stage), g.k_cache,
                    {(int)g.num_pages * prefill_info.layer_idx + kv_page_index, 0, prefill_info.kv_head_idx, 0},
                    K_arrived(s, stage));

                warp::tma::expect(V_arrived(s, stage), V(s, stage));
                warp::tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(
                    V(s, stage), g.v_cache,
                    {(int)g.num_pages * prefill_info.layer_idx + kv_page_index, 0, prefill_info.kv_head_idx, 0},
                    V_arrived(s, stage));
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
        static __device__ void run(const Globals &g, state<Config> &s) {
            static_assert(Config::NUM_CONSUMER_WARPS == 8, "Fix this function.");

            int wgid = warpgroup::groupid();
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
            prefill_instruction prefill_info(g, s);

            q_st(&Q_smem) = Q(s, wgid);
            wait(Q_arrived(s), 0);  // Await Q arrival

            for (int i = 0; i < prefill_info.attn_blocks; ++i) {
                int stage = i % NUM_STAGES;
                // Perform Q @ K.T
                warp::wait(K_arrived(s, stage), (i / NUM_STAGES) % 2);
                // s.consumer_record(COMPUTE_EVENT);
                warpgroup::mm_ABt(attn_fl_reg, Q_smem, K(s, stage));
                warpgroup::mma_async_wait();
                warp::arrive(K_finished(s, stage));

                if (i >= prefill_info.attn_blocks - NUM_STAGES) {
                    s.warp_finish_page(K_page(s, stage), 1);
                }
                if (i + 1 == prefill_info.attn_blocks) {
                    s.warp_finish_page(Q_page(s), 1);  // Finish the Q page.
                }

                // // Mask out invalid positions at the end
                // if (i == prefill_info.attn_blocks - 1) {
                //     int remaining_sequence_length =
                //         prefill_info.sequence_length - (i * kv_page_size) +
                //         (15 - min(15, prefill_info.abs_q_row_last -
                //                           prefill_info.abs_q_row));  // need to adjust if not actually full of queries.
                //     if (remaining_sequence_length < 256) {
                //         warp::apply(attn_fl_reg, attn_fl_reg,
                //                     [remaining_sequence_length] __device__(int row, int col, float val) {
                //                         return (col >= (remaining_sequence_length + row - 15)) ? -999999999999.f :
                //                         val;
                //                     });
                //     }
                // }

                auto kv_seqlen_at_block_start = i * kv_page_size;
                auto kv_seqlen_with_this_block = (i + 1) * kv_page_size;

                auto q_pos_start = prefill_info.rel_q_row + prefill_info.prefill_token_offset;

                if (kv_seqlen_with_this_block > q_pos_start) {
                    warp::apply(attn_fl_reg, attn_fl_reg,
                                [kv_seqlen_at_block_start, q_pos_start] __device__(int row, int col, float val) {
                                    auto kv_pos = kv_seqlen_at_block_start + col;

                                    // qs are packed like is
                                    // [q0h0, q1h0, q2h0, …, q0h1, q1h1, …]
                                    // so each warp contains all the 16 positions for a single head.
                                    auto q_pos = row + q_pos_start;

                                    return (kv_pos > q_pos) ? -999999999999.f : val;
                                });
                }

                // Obtain maximums per row (which is per head)
                warp::row_max(max_vec_reg, attn_fl_reg, max_vec_reg);  // includes previous max

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
                warp::copy(attn_bf_reg, attn_fl_reg);  // Convert to bf16 to do matmul

                warp::wait(V_arrived(s, stage), (i / NUM_STAGES) % 2);

                // Normalize and accumulate demoniator
                warp::mul(norm_vec_reg, norm_vec_reg, diff_scaled_max_vec_reg);
                warp::row_sum(norm_vec_reg, attn_fl_reg, norm_vec_reg);

                // Save for next iteration
                warp::copy(last_scaled_max_vec_reg, scaled_max_vec_reg);

                warpgroup::mma_AB(O_reg, attn_bf_reg, V(s, stage));
                warpgroup::mma_async_wait();

                warp::arrive(V_finished(s, stage));
                if (i >= prefill_info.attn_blocks - NUM_STAGES) {
                    s.warp_finish_page(V_page(s, stage), 1);
                }
            }
            if (prefill_info.attn_blocks < NUM_STAGES) {
                for (int i = prefill_info.attn_blocks; i < NUM_STAGES; i++) {
                    s.wait_page_ready(K_page(s, i));
                    s.wait_page_ready(V_page(s, i));
                    s.warp_finish_page(K_page(s, i), 1);
                    s.warp_finish_page(V_page(s, i), 1);
                }
            }

            // Next normalize the output according with the current logsumexp
            warp::add(norm_vec_reg, norm_vec_reg, 1e-16f);
            warp::div_row(O_reg, O_reg, norm_vec_reg);

            s.consumer_record(STORE_EVENT);

            s.wait_page_ready(O_page(s));
            warp::store(O(s, warpid()), O_reg);
            group<8>::sync(0);

            rv_fl<128> out_vecs[16];
#pragma unroll
            for (int i = 0; i < 16; i++) {
                warp::load(out_vecs[i], O(s, warpid()), {i, 0});
            }
            group<8>::sync(0);
            head_vec_sv(&O_vecs_ref)[8][16] = O_vecs(s);
#pragma unroll
            for (int i = 0; i < 16; i++) {
                warp::store(O_vecs_ref[warpid()][i], out_vecs[i]);
            }
            warp::sync();
            warp::arrive(O_arrived(s));
        }
    };
    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            int laneid = warp::laneid();
            prefill_instruction prefill_info(g, s);
            int batch_size_per_dev = g.batch_size / Globals::num_devices;
            constexpr int heads_per_dev = Globals::num_attention_heads / Globals::num_devices;

            head_vec_sv(&O_vecs_ref)[8][16] = O_vecs(s);
            wait(O_arrived(s), 0);

#pragma unroll
            for (int i = 0; i < 4; i++) {
                int thread_q_head = i * 2 + laneid / 16;
                int thread_q_row = prefill_info.abs_q_row + laneid % 16;
                if (thread_q_row <= prefill_info.abs_q_row_last) {
                    int2 local_idx_info = global_batch_idx_to_local_idx_info(g, thread_q_row);
                    int target_dev_idx = local_idx_info.x;
                    int target_row = local_idx_info.y;
                    int target_col = g.dev_idx * heads_per_dev + prefill_info.kv_head_idx * GQA_RATIO + thread_q_head;
                    tma::store_async(g.attn_out[target_dev_idx], O_vecs_ref[thread_q_head][laneid % 16],
                                     {target_row, target_col});
                }
            }

            tma::store_async_read_wait();  // Wait until it's read from SMEM
            warp::sync();
            s.storer_record(READY_EVENT);
            s.warp_finish_page(O_page(s), Config::NUM_CONSUMER_WARPS);  // Finish the page.

            tma::store_async_wait();
            warp::sync();  // ensure all writes are committed
            fence<Sem::RELEASE, Scope::SYS>();
            s.storer_record(STORE2_EVENT);

#pragma unroll
            for (int i = 0; i < 4; i++) {
                // int thread_q_head = i*2 + laneid/16;
                int thread_q_row = prefill_info.abs_q_row + laneid % 16;
                if (thread_q_row <= prefill_info.abs_q_row_last) {
                    int2 local_idx_info = global_batch_idx_to_local_idx_info(g, thread_q_row);
                    int target_dev_idx = local_idx_info.x;
                    int target_row = local_idx_info.y / Globals::matmul_batch_block_size;

                    // we dump into the decode barrier for the benefit of o proj.
                    redAdd<Sem::RELAXED, Scope::SYS>(
                        &g.Bar[target_dev_idx][{prefill_info.layer_idx, OPCODE_GQA_AttentionDecode - 1, target_row, 0}],
                        1);
                }
            }
        }
    };
};
