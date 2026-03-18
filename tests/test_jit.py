import time

import torch

from megakittens.jit.c_utils import c_int, pack_args
from megakittens.jit.cuda_utils import (
    get_kernel_from_cubin_module,
    get_sm_arch,
    initialize_cuda_context,
    launch_kernel,
    load_cubin_module,
    set_kernel_dynamic_smem,
    unload_cubin_module,
)
from megakittens.jit.nvrtc_jit import compile_source_to_cubin
from megakittens.jit.pykittens import gl, st


# ---------------------------------------------------------------------------
# Simple GEMM
# ---------------------------------------------------------------------------

SIMPLE_GEMM_SOURCE = r"""
#include "kittens.cuh"
using namespace kittens;

struct globals {
    using tile_gl = gl<float, 1, 1, -1, -1>;
    tile_gl A;
    tile_gl B;
    tile_gl C;
    int N;
};

extern "C" __global__ void kernel(const __grid_constant__ globals g) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= g.N || col >= g.N) return;
    const float* A = g.A.raw_ptr;
    const float* B = g.B.raw_ptr;
    float sum = 0.0f;
    for (int k = 0; k < g.N; ++k)
        sum += A[row * g.N + k] * B[k * g.N + col];
    g.C.raw_ptr[row * g.N + col] = sum;
}
"""


def _launch_simple_gemm(fn, A, B, C, N, stream):
    BLOCK_SIZE = 32
    global_layout = gl(dtype=torch.float32, b=1, d=1, r=-1, c=-1)
    _holder, packed = pack_args([
        (global_layout.tensor_to_gl(A), global_layout.size, global_layout.align),
        (global_layout.tensor_to_gl(B), global_layout.size, global_layout.align),
        (global_layout.tensor_to_gl(C), global_layout.size, global_layout.align),
        (c_int(N), 4, 4),
    ])
    set_kernel_dynamic_smem(fn, 0)
    launch_kernel(
        fn,
        packed,
        grid=((N + BLOCK_SIZE - 1) // BLOCK_SIZE, (N + BLOCK_SIZE - 1) // BLOCK_SIZE), 
        block=(BLOCK_SIZE, BLOCK_SIZE), 
        dynamic_smem_bytes=0, 
        stream=stream
    )


def test_simple_gemm():
    N = 128
    device_index = 0

    initialize_cuda_context(device_index)
    major, minor = get_sm_arch(device_index)
    cubin, (kernel_name,) = compile_source_to_cubin(SIMPLE_GEMM_SOURCE, (b"kernel",), major, minor)
    module = load_cubin_module(cubin)
    fn = get_kernel_from_cubin_module(module, kernel_name)

    A = torch.randn(N, N, device=f"cuda:{device_index}", dtype=torch.float32)
    B = torch.randn(N, N, device=f"cuda:{device_index}", dtype=torch.float32)
    C = torch.empty(N, N, device=f"cuda:{device_index}", dtype=torch.float32)

    stream = torch.cuda.current_stream(device_index).cuda_stream
    _launch_simple_gemm(fn, A, B, C, N, stream)
    torch.cuda.synchronize(device_index)

    C_ref = A @ B
    passed = torch.equal(C, C_ref)
    print(f"test_simple_gemm: {'Passed' if passed else 'Failed'}")

    unload_cubin_module(module)


# ---------------------------------------------------------------------------
# Optimized GEMM
# ---------------------------------------------------------------------------

optimized_gemm_config = {
    "Mb": 256, "Nb": 256, "Kb": 64,
    "SUPERGROUP_SIZE": 4,
    "OVERLAP_MMA_EPI": False,
    "LOAD_PIPE_DEPTH": 4,
    "EPI_PIPE_DEPTH": 8,
    "CLUSTER_SIZE": 2,
    "NUM_CONSUMERS": 2,
    "NUM_PRODUCERS": 1,
    "DYNAMIC_SHARED_MEMORY": 227 * 1024 - 1024,
}

OPTIMIZED_GEMM_SOURCE = r"""
#include "kittens.cuh"
using namespace kittens;

template <int _Mb, int _Nb, int _Kb, int _SUPERGROUP_SIZE, bool _OVERLAP_MMA_EPI, int _LOAD_PIPE_DEPTH, int _EPI_PIPE_DEPTH>
struct config {
    static_assert(_Mb == 256, "Mb must be 256");
    static_assert(_Nb >= 16 && _Nb <= 256 && _Nb % 16 == 0, "Nb must be 16, 32, ..., 256");
    static_assert(_Kb >= 16 && _Kb % 16 == 0, "Kb must be a multiple of 16");
    static_assert(_SUPERGROUP_SIZE >= 1 && _SUPERGROUP_SIZE <= 16, "SUPERGROUP_SIZE must be 1-16");
    static_assert(_LOAD_PIPE_DEPTH >= 1 && _LOAD_PIPE_DEPTH <= 16, "LOAD_PIPE_DEPTH must be 1-16");
    static_assert(_EPI_PIPE_DEPTH >= 1 && _EPI_PIPE_DEPTH <= 16, "EPI_PIPE_DEPTH must be 1-16");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = _Kb;
    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;

    static constexpr bool OVERLAP_MMA_EPI = _OVERLAP_MMA_EPI;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int MMA_PIPE_DEPTH = OVERLAP_MMA_EPI ? 2 : 1;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr int CLC_PIPE_DEPTH = 1;

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_CONSUMERS = OVERLAP_MMA_EPI ? 1 : 2;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int NUM_D_TILES = EPI_PIPE_DEPTH > 1 ? 2 : 1;
};
template <typename C>
struct globals {
    using a_tile = st_bf<C::Mb/2, C::Kb>;
    using b_tile = st_bf<C::Nb/2, C::Kb>;
    using d_tile = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

    a_gl a;
    b_gl b;
    d_gl d;
};

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel(const __grid_constant__ globals<C> g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.a.template prefetch_tma<typename G::a_tile>();
        g.b.template prefetch_tma<typename G::b_tile>();
        g.d.template prefetch_tma<typename G::d_tile>();
    }

    const int cta_rank = cluster_ctarank();
    const int iters_per_task = g.a.cols() / C::Kb;
    const int rblks = g.d.rows() / (C::Mb * C::NUM_CONSUMERS);
    const int cblks = g.d.cols() / C::Nb;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    typename G::a_tile (&a_smem)[C::LOAD_PIPE_DEPTH][C::NUM_CONSUMERS] = al.allocate<G::a_tile, C::LOAD_PIPE_DEPTH, C::NUM_CONSUMERS>();
    typename G::b_tile (&b_smem)[C::LOAD_PIPE_DEPTH]                   = al.allocate<G::b_tile, C::LOAD_PIPE_DEPTH>();
    typename G::d_tile (&d_smem)[C::NUM_CONSUMERS][C::NUM_D_TILES]     = al.allocate<G::d_tile, C::NUM_CONSUMERS, C::NUM_D_TILES>();

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_alloc{};
    using d_tt_t = tt<float, C::Mb/2, C::Nb>;

    __shared__ uint32_t tmem_addr;
    __shared__ clc::handle clc_handle[C::CLC_PIPE_DEPTH];
    __shared__ semaphore tmem_provisioned, tmem_finished, schedule_arrived[C::CLC_PIPE_DEPTH], schedule_finished[C::CLC_PIPE_DEPTH];
    __shared__ semaphore inputs_arrived[C::LOAD_PIPE_DEPTH], inputs_finished[C::LOAD_PIPE_DEPTH], outputs_arrived[C::NUM_CONSUMERS], outputs_finished[C::MMA_PIPE_DEPTH];
    uint32_t bitfield = 0xFFFF0000;

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(tmem_finished, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::CLC_PIPE_DEPTH; i++) {
            init_semaphore(schedule_arrived[i], 0, 1);
            init_semaphore(schedule_finished[i], 0, (2+C::NUM_CONSUMERS)*C::CLUSTER_SIZE+C::NUM_CONSUMERS);
        }
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, C::NUM_CONSUMERS);
            init_semaphore(inputs_finished[i], 0, C::NUM_CONSUMERS);
        }
        #pragma unroll
        for (int i = 0; i < C::NUM_CONSUMERS; i++) {
            init_semaphore(outputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < C::MMA_PIPE_DEPTH; i++) {
            init_semaphore(outputs_finished[i], 0, C::CLUSTER_SIZE*C::NUM_CONSUMERS);
        }
    }
    everyone::tma::cluster::arrive_aligned();

    if (warpgroup::groupid() == C::NUM_CONSUMERS) {
        warpgroup::decrease_registers<56>();

        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            int input_ring = 0;
            int2 tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, blockIdx.x/C::CLUSTER_SIZE);
            pdl::wait();
            everyone::tma::cluster::wait();
            for (int task_iter = 0; true; task_iter++) {
                for (int idx = 0; idx < iters_per_task; idx++) {
                    wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                    #pragma unroll
                    for (int i = 0; i < C::NUM_CONSUMERS; i++)
                        tma::cluster::load_async(a_smem[input_ring][i], g.a, {(tile_coord.x*2+cta_rank)*C::NUM_CONSUMERS+i, idx}, inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    tma::cluster::load_async(b_smem[input_ring], g.b, {tile_coord.y*2+cta_rank, idx}, inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    update_phasebit<1>(bitfield, input_ring);
                    input_ring=ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
                }
                wait(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], (task_iter/C::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%C::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], 0);
                if (schedule.success) tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, schedule.x/C::CLUSTER_SIZE);
                else break;
            }
        } else if (warpgroup::warpid() == 2 && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            for (int task_iter = 0; true; task_iter++) {
                if (cta_rank == 0) {
                    wait(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], ((task_iter+C::CLC_PIPE_DEPTH)/C::CLC_PIPE_DEPTH)%2);
                    clc::schedule(clc_handle[task_iter%C::CLC_PIPE_DEPTH], schedule_arrived[task_iter%C::CLC_PIPE_DEPTH]);
                }
                tma::expect_bytes(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], sizeof(clc_handle[task_iter%C::CLC_PIPE_DEPTH]));
                wait(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], (task_iter/C::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%C::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], 0);
                if (!schedule.success) break;
            }
        } else if (cta_rank == 0 && warpgroup::warpid() < C::NUM_CONSUMERS && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_alloc.set_addr(tmem_addr);
            d_tt_t d_tt[C::MMA_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < C::MMA_PIPE_DEPTH; i++) {
                if constexpr(C::Mb == 256) d_tt[i] = tm_alloc.template allocate<d_tt_t>(   (i+warpgroup::warpid())*C::Nb);
                else                       d_tt[i] = tm_alloc.template allocate<d_tt_t>(0, (i+warpgroup::warpid())*C::Nb);
            }
            int input_ring = 0;
            for (int task_iter = 0; true; task_iter++) {
                wait(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], (task_iter/C::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%C::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], 0);
                wait(outputs_finished[task_iter%C::MMA_PIPE_DEPTH], ((task_iter+C::MMA_PIPE_DEPTH)/C::MMA_PIPE_DEPTH)%2);
                for(int idx = 0; idx < iters_per_task; idx++) {
                    tma::expect_bytes(inputs_arrived[input_ring], (C::CLUSTER_SIZE*C::NUM_CONSUMERS*sizeof(G::a_tile) + 2*sizeof(G::b_tile))/C::NUM_CONSUMERS);
                    wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                    if (idx == 0) mm2_ABt (d_tt[task_iter%C::MMA_PIPE_DEPTH], a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                    else          mma2_ABt(d_tt[task_iter%C::MMA_PIPE_DEPTH], a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                    update_phasebit<0>(bitfield, input_ring);
                    input_ring=ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
                }
                detail::tcgen05::commit<C::CLUSTER_SIZE>(outputs_arrived[warpgroup::warpid()]);
                if (!schedule.success) break;
            }
        }
    }
    else {
        using epilogue_group = group<WARPGROUP_WARPS*C::NUM_CONSUMERS>;
        if constexpr (!C::OVERLAP_MMA_EPI)
            warpgroup::increase_registers<224>();
        everyone::tma::cluster::wait_aligned();
        if (epilogue_group::warpid() == 0) {
            tm_alloc.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_alloc.set_addr(tmem_addr);
        d_tt_t d_tt[C::MMA_PIPE_DEPTH];
        #pragma unroll
        for (int i = 0; i < C::MMA_PIPE_DEPTH; i++) {
            if constexpr(C::Mb == 256) d_tt[i] = tm_alloc.template allocate<d_tt_t>(   (i+warpgroup::groupid())*C::Nb);
            else                       d_tt[i] = tm_alloc.template allocate<d_tt_t>(0, (i+warpgroup::groupid())*C::Nb);
        }
        int2 tile_coord, next_tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, blockIdx.x/C::CLUSTER_SIZE);
        for(int task_iter = 0; true; task_iter++) {
            tile_coord = next_tile_coord;
            wait(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], (task_iter/C::CLC_PIPE_DEPTH)%2);
            auto schedule = clc::query(clc_handle[task_iter%C::CLC_PIPE_DEPTH]);
            warpgroup::sync(warpgroup::groupid()+1);
            warpgroup::tma::cluster::arrive(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], 0);
            if (schedule.success) next_tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, schedule.x/C::CLUSTER_SIZE);
            wait(outputs_arrived[warpgroup::groupid()], task_iter%2);
            if constexpr (C::OVERLAP_MMA_EPI) {
                rt_bf<C::Mb/8, C::Nb/C::EPI_PIPE_DEPTH> d_reg;
                #pragma unroll
                for(int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    warpgroup::load_async(d_reg, d_tt[task_iter%C::MMA_PIPE_DEPTH].template subtile<tt<float, C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                    if (i == C::EPI_PIPE_DEPTH - 1) {
                        tensor_load_wait();
                        warpgroup::sync(warpgroup::groupid()+1);
                        if (!schedule.success) warpgroup::pdl::arrive();
                        warpgroup::tma::cluster::arrive(outputs_finished[task_iter%C::MMA_PIPE_DEPTH], 0);
                    }
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::store(d_smem[warpgroup::groupid()][i%C::NUM_D_TILES], d_reg);
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.d, d_smem[warpgroup::groupid()][i%C::NUM_D_TILES], {(2*tile_coord.x+cta_rank)*C::NUM_CONSUMERS+warpgroup::groupid(), C::EPI_PIPE_DEPTH*tile_coord.y+i});
                }
            } else {
                rt_bf<C::Mb/8, C::Nb/C::EPI_PIPE_DEPTH> d_reg[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for(int i = 0; i < C::EPI_PIPE_DEPTH; i++)
                    warpgroup::load_async(d_reg[i], d_tt[task_iter%C::MMA_PIPE_DEPTH].template subtile<tt<float, C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                tensor_load_wait();
                warpgroup::sync(warpgroup::groupid()+1);
                if (!schedule.success) warpgroup::pdl::arrive();
                warpgroup::tma::cluster::arrive(outputs_finished[task_iter%C::MMA_PIPE_DEPTH], 0);
                #pragma unroll
                for(int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::store(d_smem[warpgroup::groupid()][i%C::NUM_D_TILES], d_reg[i]);
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.d, d_smem[warpgroup::groupid()][i%C::NUM_D_TILES], {(2*tile_coord.x+cta_rank)*C::NUM_CONSUMERS+warpgroup::groupid(), C::EPI_PIPE_DEPTH*tile_coord.y+i});
                }
            }
            if (!schedule.success) break;
        }
        epilogue_group::sync(4);
        if (epilogue_group::warpid() == 0) {
            if (warp::elect_leader()) tma::cluster::arrive(tmem_finished, 1-cta_rank);
            wait(tmem_finished, 0);
            tm_alloc.deprovision();
        }
    }
}
"""


def _launch_optimized_gemm(fn, A, B, D, M, N, K, stream):
    a_tile = st(dtype=torch.bfloat16, rows=optimized_gemm_config["Mb"]//2, cols=optimized_gemm_config["Kb"])
    b_tile = st(dtype=torch.bfloat16, rows=optimized_gemm_config["Nb"]//2, cols=optimized_gemm_config["Kb"])
    d_tile = st(dtype=torch.bfloat16, rows=optimized_gemm_config["Mb"]//2, cols=optimized_gemm_config["Nb"]//optimized_gemm_config["EPI_PIPE_DEPTH"])
    a_gl = gl(dtype=torch.bfloat16, b=1, d=1, r=-1, c=-1, tma_types=[a_tile])
    b_gl = gl(dtype=torch.bfloat16, b=1, d=1, r=-1, c=-1, tma_types=[b_tile])
    d_gl = gl(dtype=torch.bfloat16, b=1, d=1, r=-1, c=-1, tma_types=[d_tile])
    _holder, packed = pack_args([
        (a_gl.tensor_to_gl(A), a_gl.size, a_gl.align),
        (b_gl.tensor_to_gl(B), b_gl.size, b_gl.align),
        (d_gl.tensor_to_gl(D), d_gl.size, d_gl.align),
    ])
    launch_kernel(
        fn,
        packed,
        grid=((M//(optimized_gemm_config["NUM_CONSUMERS"]*optimized_gemm_config["Mb"]//2)) * N//optimized_gemm_config["Nb"],),
        block=((optimized_gemm_config["NUM_PRODUCERS"]+optimized_gemm_config["NUM_CONSUMERS"])*4*32,),
        dynamic_smem_bytes=optimized_gemm_config["DYNAMIC_SHARED_MEMORY"],
        stream=stream,
        cluster=(optimized_gemm_config["CLUSTER_SIZE"],)
    )


def test_optimized_gemm():
    device_index = 0
    M = N = K = 4096

    initialize_cuda_context(device_index)
    major, minor = get_sm_arch(device_index)
    kernel_expr = (f'kernel<config<{optimized_gemm_config["Mb"]}, {optimized_gemm_config["Nb"]}, {optimized_gemm_config["Kb"]}, '
                   f'{optimized_gemm_config["SUPERGROUP_SIZE"]}, {"true" if optimized_gemm_config["OVERLAP_MMA_EPI"] else "false"}, '
                   f'{optimized_gemm_config["LOAD_PIPE_DEPTH"]}, {optimized_gemm_config["EPI_PIPE_DEPTH"]}>>')
    cubin, (kernel_name,) = compile_source_to_cubin(OPTIMIZED_GEMM_SOURCE, (kernel_expr.encode("utf-8"),), major, minor)
    module = load_cubin_module(cubin)
    fn = get_kernel_from_cubin_module(module, kernel_name)
    set_kernel_dynamic_smem(fn, optimized_gemm_config["DYNAMIC_SHARED_MEMORY"])

    A = torch.randn(M, K, device=f"cuda:{device_index}", dtype=torch.bfloat16)
    B = torch.randn(N, K, device=f"cuda:{device_index}", dtype=torch.bfloat16)
    D = torch.empty(M, N, device=f"cuda:{device_index}", dtype=torch.bfloat16)

    stream = torch.cuda.current_stream(device_index).cuda_stream
    _launch_optimized_gemm(fn, A, B, D, M, N, K, stream)
    torch.cuda.synchronize(device_index)

    D_ref = A @ B.T
    passed = torch.equal(D, D_ref)
    print(f"test_optimized_gemm: {'Passed' if passed else 'Failed'}")

    unload_cubin_module(module)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tests():
    test_simple_gemm()
    test_optimized_gemm()


if __name__ == "__main__":
    run_tests()
