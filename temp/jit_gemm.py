import ctypes
import struct

import cuda.bindings.driver as cuda_driver
import torch

from cuda_utils import (
    check_cuda,
    get_kernel_from_cubin_module,
    get_sm_arch,
    initialize_cuda_context,
    load_cubin_module,
    unload_cubin_module,
    set_kernel_dynamic_smem
)
from nvrtc_jit import compile_source_to_cubin
from pykittens import gl, st

# ---------------------------------------------------------------------------
# Program logic
# ---------------------------------------------------------------------------

BLOCK_SIZE = 32


def _align_up(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) // alignment * alignment

def _pack_struct(fields: list[tuple[bytes, int, int]]) -> bytearray:
    """Pack (data, size, align) tuples into a C struct with padding."""
    off, max_align = 0, 1
    for _, sz, al in fields:
        off = _align_up(off, al)
        off += sz
        max_align = max(max_align, al)
    buf = bytearray(_align_up(off, max_align))
    off = 0
    for data, sz, al in fields:
        off = _align_up(off, al)
        buf[off:off+sz] = data
        off += sz
    return buf


def launch(fn, tile_gl: gl, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int, stream):
    grid_x = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_y = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    globals_buf = _pack_struct([
        (tile_gl.tensor_to_gl(A), tile_gl.size, tile_gl.align),
        (tile_gl.tensor_to_gl(B), tile_gl.size, tile_gl.align),
        (tile_gl.tensor_to_gl(C), tile_gl.size, tile_gl.align),
        (struct.pack('<i', N), 4, 4),
    ])
    ct_globals = (ctypes.c_char * len(globals_buf)).from_buffer(globals_buf)
    packed = (ctypes.c_void_p * 1)(ctypes.addressof(ct_globals))

    config = cuda_driver.CUlaunchConfig()
    config.gridDimX = grid_x
    config.gridDimY = grid_y
    config.gridDimZ = 1
    config.blockDimX = 32   # NUM_THREADS = 1 warp
    config.blockDimY = 1
    config.blockDimZ = 1
    config.sharedMemBytes = 100000
    config.hStream = stream
    config.numAttrs = 0
    config.attrs = []

    (err,) = cuda_driver.cuLaunchKernelEx(config, fn, packed, 0)
    check_cuda(err)

KERNEL_SOURCE = r"""
#include "kittens.cuh"
using namespace kittens;

static constexpr int BLOCK_SIZE = 32;
static constexpr int NUM_WORKERS = 1;
static constexpr int NUM_THREADS = NUM_WORKERS * kittens::WARP_THREADS;

struct matmul_globals {
    using sub_tile = st_bf<BLOCK_SIZE, BLOCK_SIZE>;
    using tile_gl = gl<bf16, 1, 1, -1, -1, sub_tile>;
    tile_gl A;
    tile_gl B;
    tile_gl C;
    int N;
};
static_assert(sizeof(matmul_globals) == 832, "matmul_globals layout mismatch");

extern "C" __global__ void kernel(const __grid_constant__ matmul_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, BLOCK_SIZE> &As = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>>();
    st_bf<BLOCK_SIZE, BLOCK_SIZE> &Bs = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>>();

    rt_bf<BLOCK_SIZE, BLOCK_SIZE> A_reg;
    rt_bf<BLOCK_SIZE, BLOCK_SIZE> B_reg;
    rt_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::rt_layout::col> B_reg_col;
    rt_fl<BLOCK_SIZE, BLOCK_SIZE> C_accum;

    int col = blockIdx.x;
    int row = blockIdx.y;

    kittens::warp::zero(C_accum);
    int num_tiles = (g.N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile) {
        kittens::warp::load(As, g.A, {0, 0, row, tile});
        kittens::warp::load(Bs, g.B, {0, 0, tile, col});
        __syncthreads();
        kittens::warp::load(A_reg, As);
        kittens::warp::load(B_reg, Bs);
        kittens::warp::swap_layout(B_reg_col, B_reg);
        __syncthreads();
        kittens::warp::mma_AB(C_accum, A_reg, B_reg_col, C_accum);
        __syncthreads();
    }
    kittens::warp::store(g.C, C_accum, {0, 0, row, col});
}
"""


def main():
    import time

    device_index = 0
    N = 128

    initialize_cuda_context(device_index)
    major, minor = get_sm_arch(device_index)

    t0 = time.perf_counter()
    cubin = compile_source_to_cubin(KERNEL_SOURCE, major, minor)
    t1 = time.perf_counter()
    print(f"Compile 1: {t1 - t0:.4f}s")

    t2 = time.perf_counter()
    cubin = compile_source_to_cubin(KERNEL_SOURCE, major, minor)
    t3 = time.perf_counter()
    print(f"Compile 2: {t3 - t2:.4f}s")

    module = load_cubin_module(cubin)
    fn = get_kernel_from_cubin_module(module, b"kernel")
    set_kernel_dynamic_smem(fn, 100000)

    tile = st(dtype=torch.bfloat16, rows=BLOCK_SIZE, cols=BLOCK_SIZE)

    A = torch.randn(N, N, device=f"cuda:{device_index}", dtype=torch.bfloat16)
    B = torch.randn(N, N, device=f"cuda:{device_index}", dtype=torch.bfloat16)
    C = torch.empty(N, N, device=f"cuda:{device_index}", dtype=torch.bfloat16)

    stream = torch.cuda.current_stream(device_index).cuda_stream
    torch.cuda.synchronize(device_index)

    tile_gl = gl(dtype=torch.bfloat16, b=1, d=1, r=-1, c=-1, tma_types=[tile])

    t2 = time.perf_counter()
    launch(fn, tile_gl, A, B, C, N, stream)
    torch.cuda.synchronize(device_index)
    t3 = time.perf_counter()
    print(f"Kernel launch + sync: {t3 - t2:.3f}s")

    # Verify against PyTorch's matmul.
    C_ref = A.float() @ B.float()
    torch.testing.assert_close(C, C_ref.bfloat16(), atol=0.5, rtol=0)
    print("Correctness check passed!")

    # Cleanup.
    unload_cubin_module(module)


if __name__ == "__main__":
    main()
