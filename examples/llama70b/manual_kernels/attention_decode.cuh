#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

namespace manual_kernels {

using namespace kittens;

// Standalone port of csrc/itypes/llama70b/attention_decode.cuh. One block per
// sequence: 8 consumer warps (one per KV head, with GQA_RATIO=8 Q heads each)
// + 1 producer warpgroup (TMA loader on warp 0, TMA storer on warp 1).
// Flash-attention style, register-only (no tmem). Three-stage K/V pipeline.

struct attention_decode_config {
    static constexpr int HIDDEN_DIM       = 8192;
    static constexpr int HEAD_DIM         = 128;
    static constexpr int NUM_Q_HEADS      = 64;
    static constexpr int NUM_KV_HEADS     = 8;
    static constexpr int GQA_RATIO        = NUM_Q_HEADS / NUM_KV_HEADS;   // 8
    static constexpr int KV_BLOCK_SIZE    = 16;
    static constexpr int PAGE_SIZE        = 128;
    static constexpr int ITERS_PER_PAGE   = PAGE_SIZE / KV_BLOCK_SIZE;    // 8
    static constexpr int NUM_STAGES       = 3;

    static constexpr int NUM_CONSUMER_WARPS = NUM_KV_HEADS;               // 8
    static constexpr int NUM_PRODUCER_WARPS = 4;                          // 1 wg
    static constexpr int NUM_WARPS          = NUM_CONSUMER_WARPS + NUM_PRODUCER_WARPS;
    static constexpr int NUM_THREADS        = NUM_WARPS * WARP_THREADS;
};

template <typename C>
struct attention_decode_globals {
    using q_row_sv  = sv_bf<C::NUM_Q_HEADS * C::HEAD_DIM>;
    using kv_st     = st_bf<C::KV_BLOCK_SIZE, C::HEAD_DIM>;
    using o_full_sv = sv_bf<C::NUM_Q_HEADS * C::HEAD_DIM>;

    using q_gl     = gl<bf16,   1,  1, -1, -1, q_row_sv>;
    using kv_gl    = gl<bf16,  -1, -1, -1, -1, tma::descriptor<kv_st, dim::DEPTH>>;
    using o_gl     = gl<bf16,   1,  1, -1, -1, o_full_sv>;
    using pos_gl   = gl<int,    1,  1,  1,  1>;
    using scale_gl = gl<float,  1,  1,  1,  1>;

    q_gl     q;
    kv_gl    k_cache;
    kv_gl    v_cache;
    pos_gl   pos_id;
    scale_gl attn_scale;
    o_gl     o;
};

// Copies of the helpers from csrc/itypes/llama70b/attention_decode.cuh; both
// are pure register-/smem-level and don't depend on megakernel state.
template <ducks::rt::row_layout RT>
__device__ static inline void attention_right_fill(
        RT &dst, const RT &src, int col_idx,
        typename base_types::packing<typename RT::dtype>::unpacked_type val = 0) {
    if (col_idx >= dst.cols) return;
    #pragma unroll
    for (int i = 0; i < dst.height; i++) {
        #pragma unroll
        for (int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                int cx = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                int cy = cx + 1;
                dst.tiles[i][j].data[k].x = (cx >= col_idx) ? val : src.tiles[i][j].data[k].x;
                dst.tiles[i][j].data[k].y = (cy >= col_idx) ? val : src.tiles[i][j].data[k].y;
            }
        }
    }
}

template <ducks::sv::all SV, ducks::rt::all RT>
__device__ static inline void store_8_rows(SV *dst, const RT &src) {
    static_assert(RT::rows == 16, "src rows must be 16.");
    static_assert(SV::length == RT::cols, "dst length must match src cols.");

    using T2 = typename RT::dtype;
    using U  = typename SV::dtype;
    using U2 = typename base_types::packing<U>::packed_type;

    uint32_t dst_ptr[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        dst_ptr[i] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[i].data[0]));
    }
    int lane         = laneid();
    int local_row    = lane / 4;
    int local_col    = lane % 4;

    if (lane < 32) {
        for (int j = 0; j < src.width; j++) {
            U2 tmp[2];
            tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[0]);
            tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[2]);
            int col = local_col * 2 + j * 16;
            move<U2>::sts(dst_ptr[local_row] + sizeof(U) * col, tmp[0]);
            move<U2>::sts(dst_ptr[local_row] + sizeof(U) * (col + 8), tmp[1]);
        }
    }
}

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void attention_decode_kernel(
        const __grid_constant__ attention_decode_globals<C> g) {
    using G         = attention_decode_globals<C>;
    using q_row_sv  = typename G::q_row_sv;
    using kv_st     = typename G::kv_st;
    using o_full_sv = typename G::o_full_sv;
    using o_sv_bf   = sv_bf<C::HEAD_DIM>;
    using q_st      = st_bf<16, C::HEAD_DIM, false>;

    using q_rt       = rt_bf<16, C::HEAD_DIM>;
    using k_rt       = rt_bf<C::KV_BLOCK_SIZE, C::HEAD_DIM>;
    using v_rt       = rt_bf<C::KV_BLOCK_SIZE, C::HEAD_DIM, ducks::rt_layout::col>;
    using attn_fl_rt = rt_fl<16, C::KV_BLOCK_SIZE>;
    using attn_bf_rt = rt_bf<16, C::KV_BLOCK_SIZE>;
    using o_rt       = rt_fl<16, C::HEAD_DIM>;
    using max_vec_rv = typename rt_fl<16, C::HEAD_DIM>::col_vec;

    const int seq_idx        = blockIdx.x;
    const int pos            = int(g.pos_id.raw_ptr[0]);
    const int valid_seq_len  = pos + 1;
    const int total_kv_iters = (valid_seq_len + C::KV_BLOCK_SIZE - 1) / C::KV_BLOCK_SIZE;
    const int B              = gridDim.x;
    const int pages_per_seq  = g.k_cache.batch() / B;
    const int seq_start_page = seq_idx * pages_per_seq;

    extern __shared__ int __shm_attn[];
    tma_swizzle_allocator al((int*)&__shm_attn[0]);

    // qo_smem holds Q first (NUM_Q_HEADS * HEAD_DIM bf16) followed by O. Allocated
    // together so the per-KV-head q_st<16, HEAD_DIM> reinterpret at offset
    // kv_head * GQA_RATIO * sizeof(o_sv_bf) can spill its bottom 8 unused rows
    // into the O region without overrunning the allocation.
    using qo_sv = sv_bf<2 * C::NUM_Q_HEADS * C::HEAD_DIM>;
    auto &qo_smem      = al.allocate<qo_sv>();
    auto &q_row_smem   = *reinterpret_cast<q_row_sv *>(&qo_smem);
    auto &o_full_smem  = *reinterpret_cast<o_full_sv *>(
        reinterpret_cast<char *>(&qo_smem) + sizeof(q_row_sv));

    auto &k_smem = al.allocate<kv_st, C::NUM_STAGES, C::NUM_KV_HEADS>();
    auto &v_smem = al.allocate<kv_st, C::NUM_STAGES, C::NUM_KV_HEADS>();

    __shared__ semaphore Q_arrived, O_arrived;
    __shared__ semaphore K_arrived[C::NUM_STAGES],  V_arrived[C::NUM_STAGES];
    __shared__ semaphore K_finished[C::NUM_STAGES], V_finished[C::NUM_STAGES];

    if (threadIdx.x == 0) {
        init_semaphore(Q_arrived, 0, 1);
        init_semaphore(O_arrived, 0, C::NUM_CONSUMER_WARPS);
        #pragma unroll
        for (int i = 0; i < C::NUM_STAGES; i++) {
            init_semaphore(K_arrived[i],  0, 1);
            init_semaphore(V_arrived[i],  0, 1);
            init_semaphore(K_finished[i], 0, C::NUM_CONSUMER_WARPS);
            init_semaphore(V_finished[i], 0, C::NUM_CONSUMER_WARPS);
        }
    }
    everyone::tma::cluster::sync();

    if (warpgroup::groupid() == 2) {  // producer warpgroup
        warpgroup::decrease_registers<56>();
        const int prod_warp = warpgroup::warpid();

        if (prod_warp == 0 && warp::elect_leader()) {
            tma::expect_bytes(Q_arrived, sizeof(q_row_sv));
            tma::load_async<cache_policy::EVICT_LAST>(
                q_row_smem, g.q, {0, 0, seq_idx, 0}, Q_arrived);

            for (int kv_iter_idx = 0; kv_iter_idx < total_kv_iters; kv_iter_idx++) {
                const int stage     = kv_iter_idx % C::NUM_STAGES;
                const int page_idx  = seq_start_page + kv_iter_idx / C::ITERS_PER_PAGE;
                const int page_iter = kv_iter_idx % C::ITERS_PER_PAGE;

                if (kv_iter_idx >= C::NUM_STAGES) {
                    wait(K_finished[stage], (kv_iter_idx / C::NUM_STAGES - 1) & 1);
                    wait(V_finished[stage], (kv_iter_idx / C::NUM_STAGES - 1) & 1);
                }
                tma::expect_bytes(K_arrived[stage], C::NUM_KV_HEADS * sizeof(kv_st));
                tma::expect_bytes(V_arrived[stage], C::NUM_KV_HEADS * sizeof(kv_st));
                #pragma unroll
                for (int kv_head_idx = 0; kv_head_idx < C::NUM_KV_HEADS; kv_head_idx++) {
                    tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(
                        k_smem[stage][kv_head_idx], g.k_cache,
                        {page_idx, page_iter, kv_head_idx, 0}, K_arrived[stage]);
                    tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(
                        v_smem[stage][kv_head_idx], g.v_cache,
                        {page_idx, page_iter, kv_head_idx, 0}, V_arrived[stage]);
                }
            }
        } else if (prod_warp == 1 && warp::elect_leader()) {
            wait(O_arrived, 0);
            tma::store_async<cache_policy::EVICT_LAST>(
                g.o, o_full_smem, coord<o_full_sv>{0, 0, seq_idx, 0});
            tma::store_async_wait();
        }
    } else {  // 8 consumer warps (warps 0..7), one per KV head
        warpgroup::increase_registers<224>();
        const int kv_head_idx = warpid();
        const float softmax_temp = g.attn_scale.raw_ptr[0] * 1.44269504089f;

        q_rt Q_reg;
        k_rt K_reg;
        v_rt V_reg;
        o_rt O_reg;
        attn_fl_rt attn_fl_reg;
        attn_bf_rt attn_bf_reg;
        max_vec_rv scaled_max_vec_reg;
        max_vec_rv last_scaled_max_vec_reg;
        max_vec_rv diff_scaled_max_vec_reg;
        max_vec_rv norm_vec_reg;

        warp::neg_infty(scaled_max_vec_reg);
        warp::neg_infty(last_scaled_max_vec_reg);
        warp::zero(norm_vec_reg);
        warp::zero(O_reg);

        wait(Q_arrived, 0);
        q_st &q_group_smem = *reinterpret_cast<q_st *>(
            reinterpret_cast<char *>(&q_row_smem) +
            kv_head_idx * C::GQA_RATIO * sizeof(o_sv_bf));
        warp::load(Q_reg, q_group_smem);

        for (int kv_iter_idx = 0; kv_iter_idx < total_kv_iters; kv_iter_idx++) {
            const int stage = kv_iter_idx % C::NUM_STAGES;

            warp::zero(attn_fl_reg);
            wait(K_arrived[stage], (kv_iter_idx / C::NUM_STAGES) & 1);
            warp::load(K_reg, k_smem[stage][kv_head_idx]);
            warp::mma_ABt(attn_fl_reg, Q_reg, K_reg, attn_fl_reg);
            warp::sync();
            warp::arrive(K_finished[stage]);

            if ((kv_iter_idx + 1) * C::KV_BLOCK_SIZE > valid_seq_len) {
                attention_right_fill(
                    attn_fl_reg, attn_fl_reg,
                    valid_seq_len % C::KV_BLOCK_SIZE,
                    base_types::constants<float>::neg_infty());
            }

            warp::mul(attn_fl_reg, attn_fl_reg, softmax_temp);
            warp::row_max(scaled_max_vec_reg, attn_fl_reg, scaled_max_vec_reg);
            warp::sub_row(attn_fl_reg, attn_fl_reg, scaled_max_vec_reg);
            warp::exp2(attn_fl_reg, attn_fl_reg);
            warp::sub(diff_scaled_max_vec_reg, last_scaled_max_vec_reg, scaled_max_vec_reg);
            warp::exp2(diff_scaled_max_vec_reg, diff_scaled_max_vec_reg);

            warp::mul_row(O_reg, O_reg, diff_scaled_max_vec_reg);
            wait(V_arrived[stage], (kv_iter_idx / C::NUM_STAGES) & 1);
            warp::load(V_reg, v_smem[stage][kv_head_idx]);
            warp::copy(attn_bf_reg, attn_fl_reg);
            warp::mma_AB(O_reg, attn_bf_reg, V_reg, O_reg);
            warp::sync();
            warp::arrive(V_finished[stage]);

            warp::mul(norm_vec_reg, norm_vec_reg, diff_scaled_max_vec_reg);
            warp::row_sum(norm_vec_reg, attn_fl_reg, norm_vec_reg);
            warp::copy(last_scaled_max_vec_reg, scaled_max_vec_reg);
        }

        warp::div_row(O_reg, O_reg, norm_vec_reg);

        const int q_head_start = kv_head_idx * C::GQA_RATIO;
        o_sv_bf *o_per_head = reinterpret_cast<o_sv_bf *>(&o_full_smem) + q_head_start;
        store_8_rows(o_per_head, O_reg);
        warp::sync();
        warp::arrive(O_arrived);
    }
}

inline void attention_decode_dispatch(
        at::Tensor q,
        at::Tensor k_cache,
        at::Tensor v_cache,
        at::Tensor pos_id,
        at::Tensor attn_scale,
        at::Tensor o) {
    using C = attention_decode_config;

    CHECK_INPUT(q); CHECK_INPUT(k_cache); CHECK_INPUT(v_cache);
    CHECK_INPUT(pos_id); CHECK_INPUT(attn_scale); CHECK_INPUT(o);

    TORCH_CHECK(q.dim() == 2 && q.size(1) == C::HIDDEN_DIM, "q must be [B, 8192]");
    TORCH_CHECK(o.sizes() == q.sizes(), "o must match q shape");
    TORCH_CHECK(k_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes(),
                "k/v cache must be [pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM]");
    TORCH_CHECK(k_cache.size(1) == C::PAGE_SIZE && k_cache.size(2) == C::NUM_KV_HEADS &&
                k_cache.size(3) == C::HEAD_DIM,
                "k cache trailing dims must be (PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)");
    TORCH_CHECK(k_cache.size(0) % q.size(0) == 0, "num_pages must be divisible by B");
    TORCH_CHECK(pos_id.dim() == 1 && pos_id.size(0) == 1, "pos_id must be [1]");
    TORCH_CHECK(attn_scale.dim() == 1 && attn_scale.size(0) == 1, "attn_scale must be [1]");
    TORCH_CHECK(q.scalar_type()          == at::ScalarType::BFloat16);
    TORCH_CHECK(o.scalar_type()          == at::ScalarType::BFloat16);
    TORCH_CHECK(k_cache.scalar_type()    == at::ScalarType::BFloat16);
    TORCH_CHECK(v_cache.scalar_type()    == at::ScalarType::BFloat16);
    TORCH_CHECK(pos_id.scalar_type()     == at::ScalarType::Int);
    TORCH_CHECK(attn_scale.scalar_type() == at::ScalarType::Float);

    const int B = static_cast<int>(q.size(0));

    attention_decode_globals<C> g{
        kittens::py::tensor_to_gl<typename attention_decode_globals<C>::q_gl>(q),
        kittens::py::tensor_to_gl<typename attention_decode_globals<C>::kv_gl>(k_cache),
        kittens::py::tensor_to_gl<typename attention_decode_globals<C>::kv_gl>(v_cache),
        kittens::py::tensor_to_gl<typename attention_decode_globals<C>::pos_gl>(pos_id),
        kittens::py::tensor_to_gl<typename attention_decode_globals<C>::scale_gl>(attn_scale),
        kittens::py::tensor_to_gl<typename attention_decode_globals<C>::o_gl>(o),
    };

    constexpr int dyn_smem = MAX_SHARED_MEMORY - 1024;
    CUDACHECK(cudaFuncSetAttribute(
        attention_decode_kernel<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem));

    attention_decode_kernel<C><<<dim3(B), dim3(C::NUM_THREADS), dyn_smem, at::cuda::getCurrentCUDAStream()>>>(g);
}

}  // namespace manual_kernels
