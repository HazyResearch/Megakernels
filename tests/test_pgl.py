"""Multi-device PGL plumbing tests: scatter, all-reduce, matmul+all-reduce.

Each test compiles a small CUDA kernel, allocates per-device tensors, launches
on every device with a per-device ``dev_idx``, and checks output against an
eager PyTorch oracle.

Validates: ``pgl`` byte packing, P2P TMA, per-context cubin load, and the
``tma::store_add_async`` cross-device reduce primitive.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import cuda.bindings.driver as cuda_driver

from megakittens.jit.c_utils import c_int, pack_args
from megakittens.jit.cuda_utils import (
    check_cuda,
    get_kernel_from_cubin_module,
    get_sm_arch,
    initialize_cuda_context,
    launch_kernel,
    load_cubin_module,
)
from megakittens.jit.nvrtc_jit import compile_source_to_cubin
from megakittens.jit.pykittens import gl, pgl, st
from megakittens.schema.dtype import DType


NUM_DEVICES = int(os.environ.get("PGL_NUM_DEVICES", "8"))


def _enable_p2p(num_devices: int) -> None:
    ctxs = []
    for i in range(num_devices):
        initialize_cuda_context(i)
        err, dev = cuda_driver.cuDeviceGet(i)
        check_cuda(err)
        err, ctx = cuda_driver.cuDevicePrimaryCtxRetain(dev)
        check_cuda(err)
        ctxs.append(ctx)
    for src in range(num_devices):
        (err,) = cuda_driver.cuCtxSetCurrent(ctxs[src])
        check_cuda(err)
        for dst in range(num_devices):
            if src == dst:
                continue
            (err,) = cuda_driver.cuCtxEnablePeerAccess(ctxs[dst], 0)
            if err not in (
                cuda_driver.CUresult.CUDA_SUCCESS,
                cuda_driver.CUresult.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
            ):
                raise RuntimeError(f"cuCtxEnablePeerAccess({src}->{dst}) failed: {err}")


def _make_pgl(rows: int, cols: int, num_devices: int) -> pgl:
    """Build a `pgl<gl<bf16, 1, 1, -1, -1, st<rows, cols>>, N, false>` type."""
    tma_type = st(dtype=DType.bf16, rows=rows, cols=cols)
    inner = gl(dtype=DType.bf16, b=1, d=1, r=-1, c=-1, tma_types=[tma_type])
    return pgl(inner=inner, num_devices=num_devices)


def _pack_globals(pgl_slots: list[tuple[pgl, list[torch.Tensor]]], dev_idx: int):
    """pgl_slots = [(pgl_ty, [per-device-tensor, ...]), ...]. Appends dev_idx int."""
    fields = []
    for pgl_ty, tensors in pgl_slots:
        bytes_ = pgl_ty.tensors_to_pgl(
            [t.data_ptr() for t in tensors],
            [(1, 1, t.shape[0], t.shape[1]) for t in tensors],
        )
        fields.append((bytes_, pgl_ty.size, pgl_ty.align))
    fields.append((c_int(int(dev_idx)), 4, 4))
    return pack_args(fields)


def _launch_all(source: str, symbol: bytes, pgl_slots, grid, block, num_devices: int) -> None:
    """Compile source, load+launch on each device with its own dev_idx, sync all."""
    major, minor = get_sm_arch(0)
    cubin, (mangled,) = compile_source_to_cubin(
        source, (symbol,), major, minor, use_file_cache=False,
    )
    holders = []  # kept alive across launches
    for dev in range(num_devices):
        with torch.cuda.device(dev):
            module = load_cubin_module(cubin)
            fn = get_kernel_from_cubin_module(module, mangled)
            holder, packed = _pack_globals(pgl_slots, dev)
            holders.append((holder, packed, module, fn))
            stream = torch.cuda.current_stream(dev).cuda_stream
            launch_kernel(
                fn, packed, grid=grid, block=block,
                dynamic_smem_bytes=0, stream=stream, cluster=None, pdl=False,
            )
    for dev in range(num_devices):
        with torch.cuda.device(dev):
            torch.cuda.synchronize()


def run_scatter(num_devices: int = NUM_DEVICES) -> None:
    """Each device adds dev_idx to its local x tile, scatters to every peer's
    y[dev_idx * LOCAL_ROWS : (dev_idx+1) * LOCAL_ROWS, :]."""
    TILE_R, TILE_C = 128, 128
    LOCAL_ROWS, COLS = 256, 256
    _enable_p2p(num_devices)

    torch.manual_seed(0)
    x_tensors, y_tensors = [], []
    for i in range(num_devices):
        with torch.cuda.device(i):
            x_tensors.append(torch.randn(LOCAL_ROWS, COLS, dtype=torch.bfloat16, device=f"cuda:{i}"))
            y_tensors.append(torch.zeros(num_devices * LOCAL_ROWS, COLS, dtype=torch.bfloat16, device=f"cuda:{i}"))

    pgl_ty = _make_pgl(TILE_R, TILE_C, num_devices)
    source = f"""
#include "kittens.cuh"
using namespace kittens;
namespace pgl_scatter {{
static_assert(sizeof({pgl_ty.cpp_type}) == {pgl_ty.size}, "pgl layout mismatch");
using tile_t = st<bf16, {TILE_R}, {TILE_C}>;
struct Globals {{ {pgl_ty.cpp_type} x_pgl; {pgl_ty.cpp_type} y_pgl; int dev_idx; }};
__global__ __launch_bounds__(128, 1)
void kernel(const __grid_constant__ Globals g) {{
    const int tile_row = blockIdx.y, tile_col = blockIdx.x, dev_idx = g.dev_idx;
    __shared__ __align__(1024) tile_t x_smem;
    __shared__ semaphore arrived;
    if (threadIdx.x == 0) {{
        init_semaphore(arrived, 1);
        tma::expect_bytes(arrived, sizeof(tile_t));
        tma::load_async(x_smem, g.x_pgl.gls[dev_idx], {{tile_row, tile_col}}, arrived);
    }}
    __syncthreads();
    wait(arrived, 0);
    // group<4>::load(rt<R,C>, st<R*4,C>) needs st.rows / rt.rows == 4.
    rt_bf<{TILE_R} / 4, {TILE_C}> x_reg;
    group<4>::load(x_reg, x_smem);
    group<4>::add(x_reg, x_reg, (bf16)((float)dev_idx));
    group<4>::store(x_smem, x_reg);
    __syncthreads();
    if (threadIdx.x == 0) {{
        const int y_row = dev_idx * ({LOCAL_ROWS} / {TILE_R}) + tile_row;
        #pragma unroll
        for (int peer = 0; peer < {num_devices}; peer++)
            tma::store_async(g.y_pgl.gls[peer], x_smem, {{y_row, tile_col}});
        tma::store_async_wait();
    }}
}}
}}
"""
    _launch_all(
        source, b"pgl_scatter::kernel",
        [(pgl_ty, x_tensors), (pgl_ty, y_tensors)],
        grid=(COLS // TILE_C, LOCAL_ROWS // TILE_R, 1),
        block=(128, 1, 1),
        num_devices=num_devices,
    )

    expected = torch.cat([(x_tensors[i] + float(i)).cpu() for i in range(num_devices)], dim=0).float()
    max_diff = max(
        float((y_tensors[d].cpu().float() - expected).abs().max().item())
        for d in range(num_devices)
    )
    print(f"[scatter] max_diff={max_diff:.6f}")
    assert max_diff < 5e-2, f"scatter mismatch: max_diff={max_diff}"


def run_allreduce(num_devices: int = NUM_DEVICES) -> None:
    """Every device TMA-store_adds its x tile into every peer's y tile.
    After N launches, y[peer] == sum_k x_k."""
    TILE_R, TILE_C = 128, 128
    ROWS, COLS = 256, 256
    _enable_p2p(num_devices)

    torch.manual_seed(1)
    x_tensors, y_tensors = [], []
    for i in range(num_devices):
        with torch.cuda.device(i):
            x_tensors.append(torch.randn(ROWS, COLS, dtype=torch.bfloat16, device=f"cuda:{i}"))
            y_tensors.append(torch.zeros(ROWS, COLS, dtype=torch.bfloat16, device=f"cuda:{i}"))

    pgl_ty = _make_pgl(TILE_R, TILE_C, num_devices)
    source = f"""
#include "kittens.cuh"
using namespace kittens;
namespace pgl_ar {{
static_assert(sizeof({pgl_ty.cpp_type}) == {pgl_ty.size}, "pgl layout mismatch");
using tile_t = st<bf16, {TILE_R}, {TILE_C}>;
struct Globals {{ {pgl_ty.cpp_type} x_pgl; {pgl_ty.cpp_type} y_pgl; int dev_idx; }};
__global__ __launch_bounds__(128, 1)
void kernel(const __grid_constant__ Globals g) {{
    const int tile_row = blockIdx.y, tile_col = blockIdx.x, dev_idx = g.dev_idx;
    __shared__ __align__(1024) tile_t x_smem;
    __shared__ semaphore arrived;
    if (threadIdx.x == 0) {{
        init_semaphore(arrived, 1);
        tma::expect_bytes(arrived, sizeof(tile_t));
        tma::load_async(x_smem, g.x_pgl.gls[dev_idx], {{tile_row, tile_col}}, arrived);
    }}
    __syncthreads();
    wait(arrived, 0);
    if (threadIdx.x == 0) {{
        #pragma unroll
        for (int peer = 0; peer < {num_devices}; peer++)
            tma::store_add_async(g.y_pgl.gls[peer], x_smem, {{tile_row, tile_col}});
        tma::store_async_wait();
    }}
}}
}}
"""
    _launch_all(
        source, b"pgl_ar::kernel",
        [(pgl_ty, x_tensors), (pgl_ty, y_tensors)],
        grid=(COLS // TILE_C, ROWS // TILE_R, 1),
        block=(128, 1, 1),
        num_devices=num_devices,
    )

    # bf16 8-way accumulation: each add re-rounds, worst case ~0.25 for N=8
    # randn-sized values. Tolerance has headroom.
    expected = sum((x_tensors[i].cpu().float() for i in range(num_devices)),
                   start=torch.zeros_like(x_tensors[0].cpu().float()))
    max_diff = max(
        float((y_tensors[d].cpu().float() - expected).abs().max().item())
        for d in range(num_devices)
    )
    print(f"[allreduce] max_diff={max_diff:.6f}")
    assert max_diff < 0.5, f"allreduce mismatch: max_diff={max_diff}"


def run_matmul_ar(num_devices: int = NUM_DEVICES) -> None:
    """Each device computes partial = A_k @ B_k (K-sharded) and store_adds into
    every peer's y. Oracle: full_A @ full_B = sum_k (A_k @ B_k)."""
    M, N, K_LOCAL = 16, 128, 16
    _enable_p2p(num_devices)

    torch.manual_seed(7)
    a_tensors, b_tensors, y_tensors = [], [], []
    for i in range(num_devices):
        with torch.cuda.device(i):
            a_tensors.append(torch.randn(M, K_LOCAL, dtype=torch.bfloat16, device=f"cuda:{i}"))
            b_tensors.append(torch.randn(K_LOCAL, N, dtype=torch.bfloat16, device=f"cuda:{i}"))
            y_tensors.append(torch.zeros(M, N, dtype=torch.bfloat16, device=f"cuda:{i}"))

    a_ty = _make_pgl(M, K_LOCAL, num_devices)
    b_ty = _make_pgl(K_LOCAL, N, num_devices)
    y_ty = _make_pgl(M, N, num_devices)
    source = f"""
#include "kittens.cuh"
using namespace kittens;
namespace pgl_mm_ar {{
static_assert(sizeof({a_ty.cpp_type}) == {a_ty.size}, "a_pgl layout mismatch");
static_assert(sizeof({b_ty.cpp_type}) == {b_ty.size}, "b_pgl layout mismatch");
static_assert(sizeof({y_ty.cpp_type}) == {y_ty.size}, "y_pgl layout mismatch");
using a_tile_t = st<bf16, {M}, {K_LOCAL}>;
using b_tile_t = st<bf16, {K_LOCAL}, {N}>;
using d_tile_t = st<bf16, {M}, {N}>;
struct Globals {{ {a_ty.cpp_type} a_pgl; {b_ty.cpp_type} b_pgl; {y_ty.cpp_type} y_pgl; int dev_idx; }};
// Per-warp MMA (wgmma is deprecated on Blackwell; tcgen05 TMEM not needed here).
__global__ __launch_bounds__(32, 1)
void kernel(const __grid_constant__ Globals g) {{
    const int dev_idx = g.dev_idx;
    __shared__ __align__(1024) a_tile_t a_smem;
    __shared__ __align__(1024) b_tile_t b_smem;
    __shared__ __align__(1024) d_tile_t d_smem;
    __shared__ semaphore a_arrived, b_arrived;
    if (warp::laneid() == 0) {{
        init_semaphore(a_arrived, 1);
        init_semaphore(b_arrived, 1);
        tma::expect_bytes(a_arrived, sizeof(a_smem));
        tma::load_async(a_smem, g.a_pgl.gls[dev_idx], {{0, 0}}, a_arrived);
        tma::expect_bytes(b_arrived, sizeof(b_smem));
        tma::load_async(b_smem, g.b_pgl.gls[dev_idx], {{0, 0}}, b_arrived);
    }}
    warp::sync();
    wait(a_arrived, 0);
    wait(b_arrived, 0);
    rt_bf<{M}, {K_LOCAL}, ducks::rt_layout::row> a_reg;
    rt_bf<{K_LOCAL}, {N}, ducks::rt_layout::col> b_reg;
    rt_fl<{M}, {N}, ducks::rt_layout::row> c_init, d_reg;
    warp::load(a_reg, a_smem);
    warp::load(b_reg, b_smem);
    warp::zero(c_init);
    warp::mma_AB(d_reg, a_reg, b_reg, c_init);
    rt_bf<{M}, {N}, ducks::rt_layout::row> d_bf;
    warp::copy(d_bf, d_reg);
    warp::store(d_smem, d_bf);
    warp::sync();
    if (warp::laneid() == 0) {{
        #pragma unroll
        for (int peer = 0; peer < {num_devices}; peer++)
            tma::store_add_async(g.y_pgl.gls[peer], d_smem, {{0, 0}});
        tma::store_async_wait();
    }}
}}
}}
"""
    _launch_all(
        source, b"pgl_mm_ar::kernel",
        [(a_ty, a_tensors), (b_ty, b_tensors), (y_ty, y_tensors)],
        grid=(1, 1, 1),
        block=(32, 1, 1),
        num_devices=num_devices,
    )

    full_a = torch.cat([a_tensors[i].cpu().float() for i in range(num_devices)], dim=1)
    full_b = torch.cat([b_tensors[i].cpu().float() for i in range(num_devices)], dim=0)
    expected = full_a @ full_b
    max_diff = max(
        float((y_tensors[d].cpu().float() - expected).abs().max().item())
        for d in range(num_devices)
    )
    print(f"[matmul_ar] max_diff={max_diff:.4f}")
    # Loose tolerance: bf16 matmul + 8-way bf16 scatter-add.
    assert max_diff < 5.0, f"matmul_ar mismatch: max_diff={max_diff}"


try:
    import pytest

    @pytest.mark.parametrize("run", [run_scatter, run_allreduce, run_matmul_ar])
    def test_pgl(run):
        if torch.cuda.device_count() < NUM_DEVICES:
            pytest.skip(f"need {NUM_DEVICES} GPUs, found {torch.cuda.device_count()}")
        run()
except ImportError:
    pass


if __name__ == "__main__":
    if torch.cuda.device_count() < NUM_DEVICES:
        raise RuntimeError(f"need {NUM_DEVICES} GPUs, found {torch.cuda.device_count()}")
    run_scatter()
    run_allreduce()
    run_matmul_ar()
