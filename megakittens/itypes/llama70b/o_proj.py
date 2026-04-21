"""OProj — faithful reference O_ProjResidual (matmul_adds.cu with reduce_scatter=false).

Each device: D_local += A_local @ B_local. Output D is declared as a PGL,
but the op writes only to `d_pgl.gls[dev_idx]` — no cross-device traffic.
The residual is whatever value D held before this op (pre-existing
hidden_states). Callers must initialize D to the residual value.

Structurally identical to the reference: in-place ``tma::store_add_async``
on the local PGL slice, no separate residual input tensor.
"""

from typing import List, Tuple

import torch

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorMeta, TensorSpec
from ...jit.pykittens import st


_PGL_NUM_DEVICES_DEFAULT = 8


@torch.library.custom_op("megakittens::oproj", mutates_args=())
def oproj_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Eager impl: local matmul. The PGL in-place accumulation is expressed
    # by declaring the output as a PGL with num_devices in the spec; actual
    # cross-device semantics are realized by MultiDispatcher at launch.
    return a @ b


@oproj_op.register_fake
def _fake(a, b):
    return torch.empty(a.shape[0], b.shape[1], dtype=a.dtype, device=a.device)


class OProj(IType):
    TILE_M = 512
    TILE_N = 256
    TILE_K = 64
    SUPERGROUP_SIZE = 8

    NUM_DEVICES = _PGL_NUM_DEVICES_DEFAULT

    @property
    def cpp_include(self) -> str:
        return "itypes/llama70b/o_proj.cuh"

    A_TMA = st(dtype=DType.bf16, rows=128, cols=64)
    B_TMA = st(dtype=DType.bf16, rows=64, cols=128)
    D_TMA = st(dtype=DType.bf16, rows=128, cols=32)

    test_cases: list[tuple] = []
    bench_cases: list[tuple] = []
    test_atol = 1e-2
    test_rtol = 1e-2

    def test_args(self, case: tuple) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError("OProj is multi-device only; see tests/test_o_proj.py")

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.TILE_M, self.TILE_K), tma_types=[self.A_TMA]),
            TensorSpec(dtype=DType.bf16, granularity=(self.TILE_K, self.TILE_N), tma_types=[self.B_TMA]),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                dtype=DType.bf16,
                granularity=(self.TILE_M, self.TILE_N),
                tma_types=[self.D_TMA],
                num_devices=self.NUM_DEVICES,  # PGL
            ),
        ]

    @staticmethod
    def _swizzled_tile_order(num_rows: int, num_cols: int, SUPERGROUP_SIZE: int) -> List[Tuple[int, int]]:
        supergroup_numel = num_rows * SUPERGROUP_SIZE
        supersection_cols = (num_cols // SUPERGROUP_SIZE) * SUPERGROUP_SIZE
        supersection_numel = num_rows * supersection_cols
        finalsection_cols = num_cols - supersection_cols
        tiles: List[Tuple[int, int]] = []
        for linear_idx in range(num_rows * num_cols):
            supergroup_idx = linear_idx // supergroup_numel
            if linear_idx < supersection_numel:
                row_idx = (linear_idx % supergroup_numel) // SUPERGROUP_SIZE
                col_idx = supergroup_idx * SUPERGROUP_SIZE + linear_idx % SUPERGROUP_SIZE
            else:
                remainder_task_id = linear_idx - supersection_numel
                row_idx = remainder_task_id // finalsection_cols
                col_idx = supersection_cols + remainder_task_id % finalsection_cols
            if supergroup_idx & 1:
                row_idx = num_rows - row_idx - 1
            tiles.append((row_idx, col_idx))
        return tiles

    def block_indices(self, src_metas, dst_metas, src_ranges, dst_ranges):
        M, K = src_metas[0].shape
        N = src_metas[1].shape[1]
        m_tiles = M // self.TILE_M
        n_tiles = N // self.TILE_N
        indices = []
        for tile_row, tile_col in self._swizzled_tile_order(m_tiles, n_tiles, self.SUPERGROUP_SIZE):
            indices.append((tile_row, tile_col))
            indices.append((tile_row, tile_col))  # cluster duplicate
        return indices

    def num_instructions(self, src_metas, dst_metas, src_ranges, dst_ranges):
        M, K = src_metas[0].shape
        N = src_metas[1].shape[1]
        return (M // self.TILE_M) * (N // self.TILE_N) * 2

    def access_regions(self, block_index, src_metas, dst_metas):
        src_regions = [[tuple((0, s) for s in m.shape)] for m in src_metas]
        dst_regions = [[tuple((0, s) for s in m.shape)] for m in dst_metas]
        return src_regions, dst_regions

    def validate(self, src_metas, dst_metas, src_ranges, dst_ranges):
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        A = src_metas[0].shape
        B = src_metas[1].shape
        D = dst_metas[0].shape
        if A[1] != B[0]:
            raise RuntimeError(f"[MegaKittens] OProj K mismatch: {A[1]} vs {B[0]}")
        if D != (A[0], B[1]):
            raise RuntimeError(f"[MegaKittens] OProj D shape mismatch: {D} vs ({A[0]}, {B[1]})")
