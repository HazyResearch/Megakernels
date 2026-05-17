#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include "matmul_pipeline.cuh"

namespace manual_kernels {

using namespace kittens;

template <int _Nb, int _LOAD_PIPE_DEPTH>
struct qkv_config : matmul_config<_Nb, _LOAD_PIPE_DEPTH> {
    static constexpr int HIDDEN_DIM   = 8192;
    static constexpr int HEAD_DIM     = 128;
    static constexpr int NUM_Q_HEADS  = 64;
    static constexpr int NUM_KV_HEADS = 8;
    static constexpr int PAGE_SIZE    = 128;
    static constexpr int Q_DIM        = NUM_Q_HEADS * HEAD_DIM;
    static constexpr int KV_DIM       = NUM_KV_HEADS * HEAD_DIM;
    static constexpr int QKV_DIM      = Q_DIM + 2 * KV_DIM;

    static_assert(HEAD_DIM == 128, "this kernel assumes head_dim=128");
    static_assert(QKV_DIM % _Nb == 0);
    static_assert(Q_DIM  % matmul_config<_Nb, _LOAD_PIPE_DEPTH>::COLS_PER_CHUNK == 0);
    static_assert(KV_DIM % matmul_config<_Nb, _LOAD_PIPE_DEPTH>::COLS_PER_CHUNK == 0);
};

template <typename C>
struct qkv_globals {
    using a_tile   = st_bf<C::Mb / 2, C::Kb>;
    using b_tile   = st_bf<C::Nb / 2, C::Kb>;
    using d_tile   = st_bf<C::Mb / 2, C::COLS_PER_CHUNK>;
    using head_vec = sv<float, C::HEAD_DIM>;
    using app_vec  = sv<int,   C::M_INST>;

    using a_gl   = gl<bf16,  1, 1, -1, -1, a_tile>;
    using b_gl   = gl<bf16,  1, 1, -1, -1, b_tile>;
    using q_gl   = gl<bf16,  1, 1, -1, -1, d_tile>;
    using cos_gl = gl<float, 1, 1, -1, -1>;
    using sin_gl = gl<float, 1, 1, -1, -1>;
    using pos_gl = gl<int,   1, 1,  1,  1>;
    using app_gl = gl<int,   1, 1,  1, -1>;
    using kv_gl  = gl<bf16, -1, -1, -1, -1>;

    a_gl   x;
    b_gl   qkv_w;
    cos_gl rope_cos;
    sin_gl rope_sin;
    pos_gl pos_id;
    app_gl append_ids;
    kv_gl  k_cache;
    kv_gl  v_cache;
    q_gl   q;
};

// row-layout rt_bf packs (col 2k, col 2k+1) per bf16x2 reg, so rope is per-register without shuffles.
template <typename C, typename RegT, typename VecT>
__device__ static inline void apply_rope_reg(
        RegT &d_reg, const VecT &cos_smem, const VecT &sin_smem, int global_col_start) {
    const int even_base = 2 * (laneid() % 4);
    #pragma unroll
    for (int j = 0; j < RegT::width; j++) {
        const int low_e  = (global_col_start + j * 16 + even_base)     % C::HEAD_DIM;
        const int high_e = (global_col_start + j * 16 + even_base + 8) % C::HEAD_DIM;
        const float cl_e = cos_smem[low_e],      sl_e = sin_smem[low_e];
        const float cl_o = cos_smem[low_e + 1],  sl_o = sin_smem[low_e + 1];
        const float ch_e = cos_smem[high_e],     sh_e = sin_smem[high_e];
        const float ch_o = cos_smem[high_e + 1], sh_o = sin_smem[high_e + 1];

        #pragma unroll
        for (int i = 0; i < RegT::height; i++) {
            #pragma unroll
            for (int k = 0; k < 2; k++) {
                auto &r = d_reg.tiles[i][j].data[k];
                const float xe = __bfloat162float(r.x);
                const float xo = __bfloat162float(r.y);
                r.x = __float2bfloat16_rn(xe * cl_e - xo * sl_e);
                r.y = __float2bfloat16_rn(xo * cl_o + xe * sl_o);
            }
            #pragma unroll
            for (int k = 2; k < 4; k++) {
                auto &r = d_reg.tiles[i][j].data[k];
                const float xe = __bfloat162float(r.x);
                const float xo = __bfloat162float(r.y);
                r.x = __float2bfloat16_rn(xe * ch_e - xo * sh_e);
                r.y = __float2bfloat16_rn(xo * ch_o + xe * sh_o);
            }
        }
    }
}

template <typename C, typename TileT, typename AppVecT, typename KVGL>
__device__ static inline void scatter_kv_tile(
        TileT &tile, const AppVecT &append_smem, KVGL &kv_gl,
        int m, int cta_rank, int cid, int global_col_start, bool is_k) {
    const int kv_col_start = global_col_start - (is_k ? C::Q_DIM : C::Q_DIM + C::KV_DIM);
    const int head_idx     = kv_col_start / C::HEAD_DIM;
    const int dim_start    = kv_col_start % C::HEAD_DIM;
    const int row_base = ((2 * m + cta_rank) * C::NUM_CONSUMERS + cid) * C::ROWS_PER_CONSUMER;
    const int wg_tid   = warpgroup::warpid() * WARP_THREADS + laneid();
    constexpr int VEC  = 8;
    #pragma unroll
    for (int elem = wg_tid;
         elem < C::ROWS_PER_CONSUMER * (C::COLS_PER_CHUNK / VEC);
         elem += WARPGROUP_WARPS * WARP_THREADS) {
        const int row        = elem / (C::COLS_PER_CHUNK / VEC);
        const int col        = (elem % (C::COLS_PER_CHUNK / VEC)) * VEC;
        const int local_row  = row_base - m * C::M_INST + row;
        const int append_idx = append_smem[local_row];
        const int page       = append_idx / C::PAGE_SIZE;
        const int offset     = append_idx % C::PAGE_SIZE;
        const uint32_t smem_addr =
            static_cast<uint32_t>(__cvta_generic_to_shared(&tile[{row, col}]));
        uint4 v;
        asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "r"(smem_addr));
        *reinterpret_cast<uint4 *>(
            &kv_gl[coord<>(page, offset, head_idx, dim_start + col)]) = v;
    }
}

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void qkv_rope_append_kernel(const __grid_constant__ qkv_globals<C> g) {
    using P        = matmul_pipeline<C>;
    using a_tile_t = typename P::a_tile_t;
    using b_tile_t = typename P::b_tile_t;
    using d_tile_t = typename P::d_tile_t;
    using d_tt_t   = typename P::d_tt_t;
    using d_reg_t  = rt_bf<C::ROWS_PER_CONSUMER / 4, C::COLS_PER_CHUNK>;

    const int cta_rank   = cluster_ctarank();
    const int num_iters  = g.x.cols() / C::Kb;
    const int cblks      = C::QKV_DIM / C::Nb;
    const int cluster_id = blockIdx.x / C::CLUSTER_SIZE;
    const int m          = cluster_id / cblks;
    const int n          = cluster_id % cblks;

    extern __shared__ int __shm_qkv[];
    tma_swizzle_allocator al((int*)&__shm_qkv[0]);

    auto &a_smem = al.allocate<a_tile_t, C::LOAD_PIPE_DEPTH, C::NUM_CONSUMERS>();
    auto &b_smem = al.allocate<b_tile_t, C::LOAD_PIPE_DEPTH>();
    // d_smem aliases a_smem[0]; safe because outputs_arrived only flips after
    // all matmul iters finish reading a/b stages.
    auto &d_smem = *reinterpret_cast<
        d_tile_t (*)[C::NUM_CONSUMERS][C::NUM_D_TILES]>(&a_smem[0][0]);
    auto &cos_smem = al.allocate<typename qkv_globals<C>::head_vec>();
    auto &sin_smem = al.allocate<typename qkv_globals<C>::head_vec>();
    auto &app_smem = al.allocate<typename qkv_globals<C>::app_vec>();

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};

    __shared__ semaphore inputs_arrived [C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived[C::NUM_CONSUMERS];
    __shared__ semaphore scratch_ready;
    uint32_t bitfield = 0xFFFF0000;

    if (threadIdx.x == 32) {
        init_semaphore(scratch_ready, 1, 0);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i],  0, C::NUM_CONSUMERS);
            init_semaphore(inputs_finished[i], 0, C::NUM_CONSUMERS);
        }
        #pragma unroll
        for (int i = 0; i < C::NUM_CONSUMERS; i++) {
            init_semaphore(outputs_arrived[i], 0, 1);
        }
    }
    everyone::tma::cluster::arrive_aligned();

    if (warpgroup::groupid() == C::NUM_CONSUMERS) {
        warpgroup::decrease_registers<56>();

        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            int input_ring = 0;
            P::producer_load(a_smem, b_smem, inputs_arrived, inputs_finished,
                             bitfield, input_ring, num_iters, cta_rank, m, n,
                             g.x, g.qkv_w);
        } else if (cta_rank == 0 && warpgroup::warpid() < C::NUM_CONSUMERS && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            d_tt_t d_tt = tm_alloc.template allocate<d_tt_t>(warpgroup::warpid() * C::Nb);
            P::launcher_mma(a_smem, b_smem, inputs_arrived, inputs_finished,
                            outputs_arrived[warpgroup::warpid()],
                            d_tt, bitfield, num_iters, warpgroup::warpid());
        }
    } else {
        const int cid = warpgroup::groupid();

        warpgroup::increase_registers<224>();
        everyone::tma::cluster::wait_aligned();

        d_tt_t d_tt = tm_alloc.template allocate<d_tt_t>(cid * C::Nb);

        if (cid == 0 && warpgroup::warpid() == 0) {
            const int tid = laneid();
            const int pos = g.pos_id.raw_ptr[0];
            const float *cos_src = g.rope_cos.raw_ptr   + pos * C::HEAD_DIM;
            const float *sin_src = g.rope_sin.raw_ptr   + pos * C::HEAD_DIM;
            const int   *app_src = g.append_ids.raw_ptr + m * C::M_INST;
            #pragma unroll
            for (int i = tid; i < C::HEAD_DIM; i += WARP_THREADS) {
                cos_smem[i] = cos_src[i];
                sin_smem[i] = sin_src[i];
            }
            #pragma unroll
            for (int i = tid; i < C::M_INST; i += WARP_THREADS) {
                app_smem[i] = app_src[i];
            }
            __syncwarp();
            if (warp::elect_leader()) warp::arrive(scratch_ready);
        }
        wait(scratch_ready, 0);
        wait(outputs_arrived[cid], 0);

        #pragma unroll
        for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
            const int slot             = i % C::NUM_D_TILES;
            const int global_chunk     = C::EPI_PIPE_DEPTH * n + i;
            const int global_col_start = global_chunk * C::COLS_PER_CHUNK;

            d_reg_t d_reg;
            warpgroup::load_async(
                d_reg,
                d_tt.template subtile<tt<float, C::ROWS_PER_CONSUMER, C::COLS_PER_CHUNK>>(
                    0, C::COLS_PER_CHUNK * i));
            tensor_load_wait();

            if (global_col_start < C::Q_DIM + C::KV_DIM) {
                apply_rope_reg<C>(d_reg, cos_smem, sin_smem, global_col_start);
            }

            warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
            warpgroup::sync(cid + 1);
            d_tile_t &out_tile = d_smem[cid][slot];
            warpgroup::store(out_tile, d_reg);
            warpgroup::sync(cid + 1);

            if (global_col_start < C::Q_DIM) {
                const int row_tile = (2 * m + cta_rank) * C::NUM_CONSUMERS + cid;
                warpgroup::tma::store_async(g.q, out_tile, {0, 0, row_tile, global_chunk});
            } else {
                const bool is_k = global_col_start < C::Q_DIM + C::KV_DIM;
                auto &kv_gl = is_k ? g.k_cache : g.v_cache;
                scatter_kv_tile<C>(out_tile, app_smem, kv_gl,
                                   m, cta_rank, cid, global_col_start, is_k);
                warpgroup::sync(cid + 1);
            }
        }

        warpgroup::tma::store_async_wait();
    }
}

inline void qkv_rope_append_dispatch(
        at::Tensor x,
        at::Tensor qkv_w,
        at::Tensor rope_cos,
        at::Tensor rope_sin,
        at::Tensor pos_id,
        at::Tensor append_ids,
        at::Tensor k_cache,
        at::Tensor v_cache,
        at::Tensor q_out) {
    using C = qkv_config</*Nb=*/256, /*LOAD_PIPE_DEPTH=*/4>;

    CHECK_INPUT(x); CHECK_INPUT(qkv_w); CHECK_INPUT(rope_cos); CHECK_INPUT(rope_sin);
    CHECK_INPUT(pos_id); CHECK_INPUT(append_ids); CHECK_INPUT(k_cache); CHECK_INPUT(v_cache);
    CHECK_INPUT(q_out);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == C::HIDDEN_DIM, "x must be [B, HIDDEN_DIM]");
    TORCH_CHECK(qkv_w.dim() == 3 && qkv_w.size(0) == 1
                && qkv_w.size(1) == C::QKV_DIM && qkv_w.size(2) == C::HIDDEN_DIM,
                "qkv_w must be [1, QKV_DIM, HIDDEN_DIM]");
    TORCH_CHECK(rope_cos.dim() == 2 && rope_cos.size(1) == C::HEAD_DIM);
    TORCH_CHECK(rope_sin.sizes() == rope_cos.sizes());
    TORCH_CHECK(pos_id.dim() == 1 && pos_id.size(0) == 1);
    TORCH_CHECK(append_ids.dim() == 1 && append_ids.size(0) == x.size(0));
    TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(1) == C::PAGE_SIZE
                && k_cache.size(2) == C::NUM_KV_HEADS && k_cache.size(3) == C::HEAD_DIM);
    TORCH_CHECK(v_cache.sizes() == k_cache.sizes());
    TORCH_CHECK(q_out.dim() == 2 && q_out.size(0) == x.size(0) && q_out.size(1) == C::Q_DIM);
    TORCH_CHECK(x.scalar_type()          == at::ScalarType::BFloat16);
    TORCH_CHECK(qkv_w.scalar_type()      == at::ScalarType::BFloat16);
    TORCH_CHECK(rope_cos.scalar_type()   == at::ScalarType::Float);
    TORCH_CHECK(rope_sin.scalar_type()   == at::ScalarType::Float);
    TORCH_CHECK(pos_id.scalar_type()     == at::ScalarType::Int);
    TORCH_CHECK(append_ids.scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(k_cache.scalar_type()    == at::ScalarType::BFloat16);
    TORCH_CHECK(v_cache.scalar_type()    == at::ScalarType::BFloat16);
    TORCH_CHECK(q_out.scalar_type()      == at::ScalarType::BFloat16);

    const int B = static_cast<int>(x.size(0));
    TORCH_CHECK(B % C::M_INST == 0, "B must be divisible by M_INST (=512)");

    qkv_globals<C> g{
        kittens::py::tensor_to_gl<typename qkv_globals<C>::a_gl  >(x),
        kittens::py::tensor_to_gl<typename qkv_globals<C>::b_gl  >(qkv_w),
        kittens::py::tensor_to_gl<typename qkv_globals<C>::cos_gl>(rope_cos),
        kittens::py::tensor_to_gl<typename qkv_globals<C>::sin_gl>(rope_sin),
        kittens::py::tensor_to_gl<typename qkv_globals<C>::pos_gl>(pos_id),
        kittens::py::tensor_to_gl<typename qkv_globals<C>::app_gl>(append_ids),
        kittens::py::tensor_to_gl<typename qkv_globals<C>::kv_gl >(k_cache),
        kittens::py::tensor_to_gl<typename qkv_globals<C>::kv_gl >(v_cache),
        kittens::py::tensor_to_gl<typename qkv_globals<C>::q_gl  >(q_out),
    };

    constexpr int dyn_smem = MAX_SHARED_MEMORY - 1024;
    CUDACHECK(cudaFuncSetAttribute(
        qkv_rope_append_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem));

    const int rblks = B / C::M_INST;
    const int cblks = C::QKV_DIM / C::Nb;
    const dim3 grid(rblks * cblks * C::CLUSTER_SIZE);
    const dim3 block(C::NUM_THREADS);
    qkv_rope_append_kernel<C><<<grid, block, dyn_smem>>>(g);
}

}  // namespace manual_kernels
