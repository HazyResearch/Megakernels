import struct
from typing import Literal

import cuda.bindings.driver as cuda_driver
from pydantic import BaseModel, Field, model_validator
import torch

from .c_utils import align_up
from .cuda_utils import check_cuda
from ..schema.dtype import DType


class st(BaseModel):
    """Slightly modified version of ThunderKittens `st`"""

    # Template parameters
    dtype: DType
    rows: int = Field(gt=0)
    cols: int = Field(gt=0)
    swizzle: bool = True
    swizzle_bytes: Literal[0, 32, 64, 128] = 0
    axis: Literal[0, 1, 2] = 2

    @model_validator(mode='after')
    def _validate_st(self):
        dtype_size = self.dtype.size
        if self.swizzle:
            assert self.rows % 16 == 0, f"rows must be divisible by 16"
            if dtype_size == 1: assert self.cols % 32 == 0, f"cols must be divisible by 32"
            else:               assert self.cols % 16 == 0, f"cols must be divisible by 16"
        else:
            assert self.cols % 16 == 0, f"cols must be divisible by 16"
        if self.swizzle_bytes == 0 and self.swizzle:
            ratio = self.cols // (32 if dtype_size == 1 else 16)
            if dtype_size <= 2:
                self.swizzle_bytes = 128 if (ratio % 4 == 0) else (64 if (ratio % 2 == 0) else 32)
            else:
                self.swizzle_bytes = 128 if ratio % 2 == 0 else 64
        return self

    @property
    def cpp_type(self) -> str:
        """C++ type for use as TMA descriptor parameter in gl template."""
        swizzle_str = "true" if self.swizzle else "false"
        st_type = f"kittens::st<{self.dtype.cpp_dtype}, {self.rows}, {self.cols}, {swizzle_str}>"
        return f"kittens::tma::descriptor<{st_type}, {self.axis}>"


class sv(BaseModel):
    """Python mirror of ThunderKittens `sv`"""

    # Template parameters
    dtype: DType
    length: int = Field(gt=0)

    @model_validator(mode='after')
    def _validate_sv(self):
        assert self.length % 16 == 0, "length must be divisible by 16"
        return self

    @property
    def cpp_type(self) -> str:
        """C++ type for use as TMA descriptor parameter in gl template."""
        sv_type = f"kittens::sv<{self.dtype.cpp_dtype}, {self.length}>"
        return f"kittens::tma::descriptor<{sv_type}, -1>"


class gl(BaseModel):
    """Python mirror of ThunderKittens `gl`."""

    # Template parameters
    dtype: DType
    b: int                                            # >0 compile dim, -1 runtime dim
    d: int
    r: int
    c: int
    tma_types: list[st | sv] = []

    @model_validator(mode='after')
    def _validate_gl(self):
        for name, s in [('b', self.b), ('d', self.d), ('r', self.r), ('c', self.c)]:
            assert s > 0 or s == -1, f"{name} must be >0 or -1"
        return self

    @property
    def cpp_type(self) -> str:
        base = f"kittens::gl<{self.dtype.cpp_dtype}, {self.b}, {self.d}, {self.r}, {self.c}"
        if self.tma_types:
            tma_args = ", ".join(t.cpp_type for t in self.tma_types)
            return f"{base}, {tma_args}>"
        else:
            return f"{base}>"

    @property
    def align(self) -> int:
        return 64 if self.tma_types else 8  # CUtensorMap: alignas(64) in NVRTC

    @property
    def memory_layout(self):
        offset = 8  # raw_ptr
        field_offsets = {'raw_ptr': 0}
        for name, s in [('batch', self.b), ('depth', self.d), ('rows', self.r), ('cols', self.c)]:
            if s > 0: # compiled_dim: empty struct, 1 byte
                field_offsets[name] = offset
                offset += 1
            else:     # runtime_dim: size_t, 8 bytes
                offset = align_up(offset, 8)
                field_offsets[name] = offset
                offset += 8
        if len(self.tma_types) > 0:
            offset = align_up(offset, 64)
            for i in range(len(self.tma_types)):
                field_offsets[f'tma_desc_{i}'] = offset
                offset += 128
            offset += 1  # empty descriptor_dict<> tail
            total_size = align_up(offset, 64)
        else:
            offset += 1  # empty descriptor_dict<> tail
            total_size = align_up(offset, 8)
        return {
            "total_size": total_size,
            "field_offsets": field_offsets
        }

    @property
    def size(self) -> int:
        return self.memory_layout["total_size"]

    def create_tma_descriptor(
        self,
        data_ptr: int,
        batch: int, depth: int, rows: int, cols: int,
        tma_type: st | sv,
    ) -> bytes:
        dtype_size = tma_type.dtype.size
        tma_format = tma_type.dtype.tma_dtype

        if isinstance(tma_type, sv):
            assert tma_type.length <= 256 or (tma_type.length * dtype_size) % 128 == 0
            dim = 16
            for d in range(16, 0, -1):
                _dim = 16*d
                if tma_type.length%_dim == 0 and (tma_type.length < 256 or (_dim*dtype_size)%128 == 0):
                    dim = _dim
                    break
            tma_dim = 4
            tma_swizzle = cuda_driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
            gmem_shape = [cols, rows, depth, batch]
            gmem_stride = [cols*dtype_size, cols*rows*dtype_size, cols*rows*depth*dtype_size]
            smem_shape = [dim, 1, 1, 1]
            smem_stride = [1] * tma_dim

        elif isinstance(tma_type, st):
            tma_swizzle = {
                0: cuda_driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
                32: cuda_driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
                64: cuda_driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
                128: cuda_driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
            }[tma_type.swizzle_bytes if tma_type.swizzle else 0]

            swizzle_elements = tma_type.swizzle_bytes // dtype_size
            assert tma_type.axis in {0, 1, 2}, "axis must be 0, 1, or 2"
            if tma_type.dtype == DType.fp4e2m1x2:
                assert tma_type.axis == 2, "Axes 0 and 1 are not yet supported for FP4 type"

            if tma_type.swizzle and tma_type.axis == 2:
                tma_dim = 5
                gmem_shape = [swizzle_elements, rows, (cols+swizzle_elements-1)//swizzle_elements, depth, batch]
                gmem_stride = [cols*dtype_size, tma_type.swizzle_bytes, rows*cols*dtype_size, depth*rows*cols*dtype_size]
                smem_shape = [swizzle_elements, tma_type.rows, tma_type.cols//swizzle_elements, 1, 1]
                smem_stride = [1] * tma_dim
            elif tma_type.swizzle and tma_type.axis == 1:
                tma_dim = 5
                gmem_shape = [swizzle_elements, depth, (cols+swizzle_elements-1)//swizzle_elements, rows, batch]
                gmem_stride = [rows*cols*dtype_size, tma_type.swizzle_bytes, cols*dtype_size, depth*rows*cols*dtype_size]
                smem_shape = [swizzle_elements, tma_type.rows, tma_type.cols//swizzle_elements, 1, 1]
                smem_stride = [1] * tma_dim
            elif tma_type.swizzle and tma_type.axis == 0:
                tma_dim = 5
                gmem_shape = [swizzle_elements, batch, (cols+swizzle_elements-1)//swizzle_elements, rows, depth]
                gmem_stride = [depth*rows*cols*dtype_size, tma_type.swizzle_bytes, cols*dtype_size, rows*cols*dtype_size]
                smem_shape = [swizzle_elements, tma_type.rows, tma_type.cols//swizzle_elements, 1, 1]
                smem_stride = [1] * tma_dim
            else:
                assert tma_type.axis == 2, "non-swizzled only supports axis=2"
                tma_dim = 4
                gmem_shape = [cols, rows, depth, batch]
                gmem_stride = [cols*dtype_size, rows*cols*dtype_size, depth*rows*cols*dtype_size]
                smem_shape = [tma_type.cols, tma_type.rows, 1, 1]
                smem_stride = [1] * tma_dim

        else:
            raise RuntimeError("[MegaKittens] Invalid tma_type")

        assert data_ptr & 0xF == 0, "memory address must be 16-byte aligned"
        for i in range(len(gmem_stride)):
            assert gmem_stride[i] % 16 == 0, f"gmem_stride[{i}] must be a multiple of 16 bytes"
        for i in range(min(3, len(smem_shape))):
            assert smem_shape[i] <= 256, f"smem_shape[{i}] must be <= 256"
        assert (smem_shape[0] * dtype_size) % 16 == 0, "smem_shape[0] * dtype_size must be a multiple of 16 bytes"
        for i in range(len(smem_stride)):
            assert smem_stride[i] <= 8, f"smem_stride[{i}] must be <= 8"
        assert smem_stride[0] == 1, "smem_stride[0] must be 1"
        if isinstance(tma_type, st) and tma_type.swizzle:
            assert smem_shape[0] * dtype_size <= tma_type.swizzle_bytes, "smem_shape[0] * dtype_size must be <= swizzle_bytes"

        err, tmap = cuda_driver.cuTensorMapEncodeTiled(
            tma_format, tma_dim, data_ptr,
            [cuda_driver.cuuint64_t(x) for x in gmem_shape], [cuda_driver.cuuint64_t(x) for x in gmem_stride],
            [cuda_driver.cuuint32_t(x) for x in smem_shape], [cuda_driver.cuuint32_t(x) for x in smem_stride],
            cuda_driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            tma_swizzle,
            cuda_driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
            cuda_driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
        check_cuda(err)
        return struct.pack('<16Q', *(int(x) for x in tmap.opaque))

    def tensor_to_gl_bytes(self, data_ptr: int, shape: tuple[int, int, int, int]) -> bytes:
        """Pack a gl with an explicit device pointer and shape (no torch.Tensor required).

        Used by `pgl.tensors_to_pgl` to build each per-device gl header.
        """
        assert self.b == -1 or shape[0] == self.b, f"Batch mismatch: expected {self.b}, got {shape[0]}"
        assert self.d == -1 or shape[1] == self.d, f"Depth mismatch: expected {self.d}, got {shape[1]}"
        assert self.r == -1 or shape[2] == self.r, f"Row mismatch: expected {self.r}, got {shape[2]}"
        assert self.c == -1 or shape[3] == self.c, f"Col mismatch: expected {self.c}, got {shape[3]}"
        layout = self.memory_layout
        buf = bytearray(layout["total_size"])
        struct.pack_into('<Q', buf, layout["field_offsets"]['raw_ptr'], data_ptr)
        for name, s, v in [('batch', self.b, shape[0]), ('depth', self.d, shape[1]),
                           ('rows',  self.r, shape[2]), ('cols',  self.c, shape[3])]:
            if s == -1:
                struct.pack_into('<Q', buf, layout["field_offsets"][name], v)
        tma_descs = [self.create_tma_descriptor(data_ptr, *shape, tma_type) for tma_type in self.tma_types]
        for i, desc in enumerate(tma_descs):
            offset = layout["field_offsets"][f'tma_desc_{i}']
            buf[offset:offset+128] = desc
        return bytes(buf)

    def tensor_to_gl(self, t: torch.Tensor) -> bytes:
        assert t.is_cuda, "Tensor must be on CUDA device"
        assert t.is_contiguous(), "Tensor must be contiguous"
        assert t.ndim <= 4, "Expected tensor.ndim <= 4"
        assert t.dtype == self.dtype.torch_dtype, f"dtype mismatch: expected {self.dtype}, got {t.dtype}"

        shape = [1, 1, 1, 1]
        for i in range(t.ndim):
            shape[4 - t.ndim + i] = t.shape[i]

        assert self.b == -1 or shape[0] == self.b, f"Batch mismatch: expected {self.b}, got {shape[0]}"
        assert self.d == -1 or shape[1] == self.d, f"Depth mismatch: expected {self.d}, got {shape[1]}"
        assert self.r == -1 or shape[2] == self.r, f"Row mismatch: expected {self.r}, got {shape[2]}"
        assert self.c == -1 or shape[3] == self.c, f"Col mismatch: expected {self.c}, got {shape[3]}"

        # Pack into C++ struct layout
        layout = self.memory_layout
        buf = bytearray(layout["total_size"])
        struct.pack_into('<Q', buf, layout["field_offsets"]['raw_ptr'], t.data_ptr())
        for name, s, v in [('batch', self.b, shape[0]), ('depth', self.d, shape[1]),
                           ('rows',  self.r, shape[2]), ('cols',  self.c, shape[3])]:
            if s == -1:
                struct.pack_into('<Q', buf, layout["field_offsets"][name], v)
        tma_descs = [self.create_tma_descriptor(t.data_ptr(), *shape, tma_type) for tma_type in self.tma_types]
        for i, desc in enumerate(tma_descs):
            offset = layout["field_offsets"][f'tma_desc_{i}']
            buf[offset:offset+128] = desc
        return bytes(buf)


class pgl(BaseModel):
    """Python mirror of ThunderKittens `pgl` — a parallel global layout spread across
    ``num_devices`` peer GPUs.

    Packs the same bytes the kernel expects for a
    ``kittens::pgl<GL, NUM_DEVICES, INIT_MC=false, INIT_TMA=false, TMA_Types...>`` struct.

    We deliberately default ``INIT_MC=false`` and ``INIT_TMA=false`` because the runtime
    kernel we're building doesn't use collective multicast ops — every TMA reference is
    via the per-device ``gls[dev_idx]``. This lets us skip the ``cuMulticastCreate`` /
    ``cuMulticastBind`` dance entirely; the kernel-visible state we populate is
    ``gls[N]`` + ``device_ids[N]`` + zero-padded multicast/TMA-dict slots.

    The owning ``gl`` configuration is specified via ``inner`` and is shared across all
    N devices (all slices have identical shape/dtype/TMA descriptor types).
    """

    inner: gl
    num_devices: int = Field(gt=1)

    @property
    def cpp_type(self) -> str:
        # megakittens::pgl_simple<GL, NUM_DEVICES> — see csrc/pgl.cuh
        gl_base = (
            f"kittens::gl<{self.inner.dtype.cpp_dtype}, {self.inner.b}, {self.inner.d}, "
            f"{self.inner.r}, {self.inner.c}"
        )
        if self.inner.tma_types:
            inner_tma = ", ".join(t.cpp_type for t in self.inner.tma_types)
            gl_base = f"{gl_base}, {inner_tma}>"
        else:
            gl_base = f"{gl_base}>"
        return f"megakittens::pgl_simple<{gl_base}, {self.num_devices}>"

    @property
    def align(self) -> int:
        # The GL's own alignment dominates (TMA descriptors inside GL are 64-byte aligned).
        return self.inner.align

    @property
    def memory_layout(self):
        """Byte layout of ``megakittens::pgl_simple<GL, N>`` (see csrc/pgl.cuh).

        Layout:
            GL gls[N];
            unsigned long long mc_size;    // unused, zero-padded
            unsigned long long mc_handle;  // unused, zero-padded
            T *mc_vas[N];                  // unused, zero-padded
            int device_ids[N];
        """
        N = self.num_devices
        gl_layout = self.inner.memory_layout
        gl_size = gl_layout["total_size"]

        field_offsets = {}
        offset = 0

        # gls[N]
        field_offsets['gls'] = offset
        offset += N * gl_size

        # mc_size (unsigned long long, 8B)
        offset = align_up(offset, 8)
        field_offsets['mc_size'] = offset
        offset += 8

        # mc_handle (unsigned long long, 8B)
        offset = align_up(offset, 8)
        field_offsets['mc_handle'] = offset
        offset += 8

        # mc_vas[N] (T*)
        offset = align_up(offset, 8)
        field_offsets['mc_vas'] = offset
        offset += 8 * N

        # device_ids[N] (int)
        offset = align_up(offset, 4)
        field_offsets['device_ids'] = offset
        offset += 4 * N

        # Align total size to the GL's own alignment.
        total_size = align_up(offset, self.inner.align)

        return {
            "total_size": total_size,
            "field_offsets": field_offsets,
            "gl_size": gl_size,
        }

    @property
    def size(self) -> int:
        return self.memory_layout["total_size"]

    def tensors_to_pgl(
        self,
        per_device_data_ptrs: list[int],
        per_device_shapes: list[tuple[int, int, int, int]],
        device_ids: list[int],
    ) -> bytes:
        """Build the byte-packed ``kittens::pgl<...>`` from explicit device pointers.

        Caller is responsible for:
        - allocating per-device tensors
        - enabling P2P access across all pairs (so peer pointers are dereferenceable)
        - passing ``per_device_data_ptrs[i]`` == ``tensors[i].data_ptr()`` on device ``device_ids[i]``

        All gls share the ``inner`` layout (dtype, shape, tma_types), so each slice must
        have the same shape. Per-device TMA descriptors are built with that device's own
        pointer — cross-device TMA access is via ``pgl.gls[peer_idx].tma_desc_0`` which
        references ``per_device_data_ptrs[peer_idx]``.

        Returns the bytes corresponding to one ``pgl`` struct ready to be copied into
        a device-global ``MKGlobals`` buffer.
        """
        N = self.num_devices
        if len(per_device_data_ptrs) != N:
            raise ValueError(f"expected {N} data pointers, got {len(per_device_data_ptrs)}")
        if len(per_device_shapes) != N:
            raise ValueError(f"expected {N} shapes, got {len(per_device_shapes)}")
        if len(device_ids) != N:
            raise ValueError(f"expected {N} device ids, got {len(device_ids)}")

        layout = self.memory_layout
        buf = bytearray(layout["total_size"])

        # gls[i] — delegate to inner.tensor_to_gl_bytes for each
        gl_size = layout["gl_size"]
        gls_offset = layout["field_offsets"]['gls']
        for i in range(N):
            gl_bytes = self.inner.tensor_to_gl_bytes(per_device_data_ptrs[i], per_device_shapes[i])
            if len(gl_bytes) != gl_size:
                raise RuntimeError(f"gl bytes size mismatch: expected {gl_size}, got {len(gl_bytes)}")
            buf[gls_offset + i * gl_size : gls_offset + (i + 1) * gl_size] = gl_bytes

        # mc_size = 0, mc_handle = 0 (we use INIT_MC=false), mc_vas[i] = 0 — leave zero-initialized

        # device_ids[N]
        dids_offset = layout["field_offsets"]['device_ids']
        for i in range(N):
            struct.pack_into('<i', buf, dids_offset + i * 4, int(device_ids[i]))

        return bytes(buf)
