import ctypes
import struct

import torch
from typing import Literal
from pydantic import BaseModel, Field, model_validator

import cuda.bindings.driver as cuda_driver

from cuda_utils import (
    check_cuda,
    get_kernel_from_cubin_module,
    get_sm_arch,
    initialize_cuda_context,
    load_cubin_module,
    unload_cubin_module,
)
from nvrtc_jit import compile_source_to_cubin

# ---------------------------------------------------------------------------
# NVRTC compilation logic
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# gl struct: Python mirror of TK's gl<T, b, d, r, c, ...TMA_Types>
# ---------------------------------------------------------------------------


def _align_up(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) // alignment * alignment


_TMA_DTYPE_MAP = {
    'bf16': (2, 'CU_TENSOR_MAP_DATA_TYPE_BFLOAT16'),
    'f16':  (2, 'CU_TENSOR_MAP_DATA_TYPE_FLOAT16'),
    'f32':  (4, 'CU_TENSOR_MAP_DATA_TYPE_FLOAT32'),
}
_TILE_COL_DIM = {1: 32, 2: 16, 4: 16}
_BASE_TILE_DIM = 16
_TILE_ROW_DIM = 16


class SharedTileInfo(BaseModel):
    """Mirrors st<T, rows, cols, swizzle, swizzle_bytes>."""
    rows: int = Field(gt=0)
    cols: int = Field(gt=0)
    dtype: Literal['bf16', 'f16', 'f32']
    swizzle: bool = True
    swizzle_bytes: Literal[0, 32, 64, 128] = 0
    axis: Literal[0, 1, 2] = 2

    @model_validator(mode='after')
    def _check_st_static_asserts(self):
        dsz = _TMA_DTYPE_MAP[self.dtype][0]
        tcol = _TILE_COL_DIM[dsz]
        # st.cuh:79 — rows % TILE_ROW_DIM == 0 (when swizzle)
        if self.swizzle:
            assert self.rows % _TILE_ROW_DIM == 0, \
                f"rows must be divisible by {_TILE_ROW_DIM}"
            assert self.cols % tcol == 0, \
                f"cols must be divisible by {tcol}"
        else:
            assert self.cols % _BASE_TILE_DIM == 0, \
                f"cols must be divisible by {_BASE_TILE_DIM}"
        # auto-compute swizzle_bytes (st.cuh:91-102)
        if self.swizzle_bytes == 0 and self.swizzle:
            r = self.cols // tcol
            if dsz <= 2:
                self.swizzle_bytes = 128 if r % 4 == 0 else 64 if r % 2 == 0 else 32
            else:
                self.swizzle_bytes = 128 if r % 2 == 0 else 64
        return self


def _tma_swizzle_enum(swizzle_bytes: int):
    return getattr(cuda_driver.CUtensorMapSwizzle, {
        0: 'CU_TENSOR_MAP_SWIZZLE_NONE', 32: 'CU_TENSOR_MAP_SWIZZLE_32B',
        64: 'CU_TENSOR_MAP_SWIZZLE_64B', 128: 'CU_TENSOR_MAP_SWIZZLE_128B',
    }[swizzle_bytes])


def create_tma_descriptor(
    data_ptr: int, batch: int, depth: int,
    rows: int, cols: int, tile: SharedTileInfo,
) -> bytes:
    """Create a 128-byte CUtensorMap (tma.cuh:create_tensor_map)."""
    dsz = _TMA_DTYPE_MAP[tile.dtype][0]
    tma_fmt_name = _TMA_DTYPE_MAP[tile.dtype][1]
    tma_fmt = getattr(cuda_driver.CUtensorMapDataType, tma_fmt_name)
    se = tile.swizzle_bytes // dsz  # swizzle_elements

    if tile.swizzle and tile.axis == 2:
        dim = 5
        gs = [se, rows, (cols+se-1)//se, depth, batch]
        gst = [cols*dsz, tile.swizzle_bytes, rows*cols*dsz, depth*rows*cols*dsz]
        ss = [se, tile.rows, tile.cols//se, 1, 1]
    elif tile.swizzle and tile.axis == 1:
        dim = 5
        gs = [se, depth, (cols+se-1)//se, rows, batch]
        gst = [rows*cols*dsz, tile.swizzle_bytes, cols*dsz, depth*rows*cols*dsz]
        ss = [se, tile.rows, tile.cols//se, 1, 1]
    elif tile.swizzle and tile.axis == 0:
        dim = 5
        gs = [se, batch, (cols+se-1)//se, rows, depth]
        gst = [depth*rows*cols*dsz, tile.swizzle_bytes, cols*dsz, rows*cols*dsz]
        ss = [se, tile.rows, tile.cols//se, 1, 1]
    else:
        assert tile.axis == 2, "non-swizzled only supports axis=2"
        dim = 4
        gs = [cols, rows, depth, batch]
        gst = [cols*dsz, rows*cols*dsz, depth*rows*cols*dsz]
        ss = [tile.cols, tile.rows, 1, 1]

    u64 = cuda_driver.cuuint64_t
    u32 = cuda_driver.cuuint32_t
    err, tmap = cuda_driver.cuTensorMapEncodeTiled(
        tma_fmt, dim, data_ptr,
        [u64(x) for x in gs], [u64(x) for x in gst],
        [u32(x) for x in ss], [u32(1)] * dim,
        cuda_driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        _tma_swizzle_enum(tile.swizzle_bytes if tile.swizzle else 0),
        cuda_driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        cuda_driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    check_cuda(err)
    return struct.pack('<16Q', *(int(x) for x in tmap.opaque))



class gl(BaseModel):
    """Python mirror of TK gl<T, b, d, r, c, ...TMA_Types>."""

    # Template parameters
    b: int  # >0 compiled, -1 runtime
    d: int
    r: int
    c: int

    # Runtime values
    raw_ptr: int
    batch: int
    depth: int
    rows: int
    cols: int
    tma_descs: list[bytes] = []

    @model_validator(mode='after')
    def _validate_gl(self):
        assert self.raw_ptr > 0, "raw_ptr cannot be null"
        for name, s, v in [('b', self.b, self.batch), ('d', self.d, self.depth),
                           ('r', self.r, self.rows),  ('c', self.c, self.cols)]:
            assert s > 0 or s == -1, f"{name} must be >0 or -1"
            if s > 0:
                assert v == s, f"{name} value {v} != static {s}"
        for desc in self.tma_descs:
            assert len(desc) == 128, f"TMA descriptor should be 128 bytes"
        return self

    @property
    def align(self) -> int:
        return 64 if self.tma_descs else 8  # CUtensorMap: alignas(64) in NVRTC

    @property
    def memory_layout(self):
        off = 8  # raw_ptr
        field_offsets = {'raw_ptr': 0}
        for name, s in [('batch', self.b), ('depth', self.d), ('rows', self.r), ('cols', self.c)]:
            if s > 0: # compiled_dim: empty struct, 1 byte
                field_offsets[name] = off
                off += 1
            else:     # runtime_dim: size_t, 8 bytes
                off = _align_up(off, 8)
                field_offsets[name] = off
                off += 8
        if len(self.tma_descs) > 0:
            off = _align_up(off, 64)
            for i in range(len(self.tma_descs)):
                field_offsets[f'tma_desc_{i}'] = off
                off += 128
            off += 1  # empty descriptor_dict<> tail
            total_size = _align_up(off, 64)
        else:
            off += 1  # empty descriptor_dict<> tail
            total_size = _align_up(off, 8)
        return {
            "total_size": total_size,
            "field_offsets": field_offsets
        }

    @property
    def size(self) -> int:
        return self.memory_layout["total_size"]

    def to_bytes(self) -> bytes:
        memory_layout = self.memory_layout
        buf = bytearray(memory_layout["total_size"])
        struct.pack_into('<Q', buf, memory_layout["field_offsets"]['raw_ptr'], self.raw_ptr)
        for name, s, v in [('batch', self.b, self.batch), ('depth', self.d, self.depth),
                           ('rows',  self.r, self.rows),  ('cols',  self.c, self.cols)]:
            if s == -1:  # only pack if runtime dimension
                struct.pack_into('<Q', buf, memory_layout["field_offsets"][name], v)
        for i, desc in enumerate(self.tma_descs):
            off = memory_layout["field_offsets"][f'tma_desc_{i}']
            buf[off:off+128] = desc
        return bytes(buf)

    @classmethod
    def from_tensor(cls, t: torch.Tensor, b=1, d=1, r=-1, c=-1,
                    tiles: list[SharedTileInfo] | None = None):
        shape = [1, 1, 1, 1]
        for i in range(t.ndim):
            shape[4 - t.ndim + i] = t.shape[i]
        descs = []
        if tiles:
            for ti in tiles:
                descs.append(create_tma_descriptor(
                    t.data_ptr(), shape[0], shape[1], shape[2], shape[3], ti))
        return cls(raw_ptr=t.data_ptr(),
                   batch=shape[0], depth=shape[1], rows=shape[2], cols=shape[3],
                   b=b, d=d, r=r, c=c, tma_descs=descs)


# ---------------------------------------------------------------------------
# Program logic
# ---------------------------------------------------------------------------

BLOCK_SIZE = 32


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


def launch(fn, gl_A: gl, gl_B: gl, gl_C: gl, N: int, stream):
    grid_x = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_y = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    globals_buf = _pack_struct([
        (gl_A.to_bytes(), gl_A.size, gl_A.align),
        (gl_B.to_bytes(), gl_B.size, gl_B.align),
        (gl_C.to_bytes(), gl_C.size, gl_C.align),
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
    (err,) = cuda_driver.cuFuncSetAttribute(
        fn, cuda_driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        100000)
    check_cuda(err)

    tile = SharedTileInfo(rows=BLOCK_SIZE, cols=BLOCK_SIZE, dtype='bf16')

    A = torch.randn(N, N, device=f"cuda:{device_index}", dtype=torch.bfloat16)
    B = torch.randn(N, N, device=f"cuda:{device_index}", dtype=torch.bfloat16)
    C = torch.empty(N, N, device=f"cuda:{device_index}", dtype=torch.bfloat16)

    stream = torch.cuda.current_stream(device_index).cuda_stream
    torch.cuda.synchronize(device_index)

    gl_A = gl.from_tensor(A, tiles=[tile])
    gl_B = gl.from_tensor(B, tiles=[tile])
    gl_C = gl.from_tensor(C, tiles=[tile])

    t2 = time.perf_counter()
    launch(fn, gl_A, gl_B, gl_C, N, stream)
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
