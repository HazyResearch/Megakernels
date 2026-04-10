import operator
from typing import List, Tuple

import torch

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorSpec
from ..jit.pykittens import st


@torch.library.custom_op("megakittens::gemm", mutates_args=())
def gemm_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b

@gemm_op.register_fake
def _gemm_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty(a.shape[0], b.shape[1], dtype=a.dtype, device=a.device)


class Gemm(IType):
    TILE_M = 512
    TILE_N = 256
    TILE_K = 64
    SUPERGROUP_SIZE = 8

    torch_functions = [
        torch.matmul, torch.mm, operator.matmul,
        torch.ops.aten.mm, torch.ops.aten.mm.default,
        torch.ops.aten.matmul, torch.ops.aten.matmul.default,
    ]
    torch_methods = ["gemm"]

    A_TMA = st(dtype=DType.bf16, rows=128, cols=64) # st_bf<Mb/2, Kb>
    B_TMA = st(dtype=DType.bf16, rows=64, cols=128) # st_bf<Kb, Nb/2> (B is K×N)
    D_TMA = st(dtype=DType.bf16, rows=128, cols=32) # st_bf<Mb/2, Nb/EPI_PIPE_DEPTH>

    test_cases = [((), (512, 256, 64)), ((), (512, 256, 256)), ((), (512, 512, 256)), ((), (1024, 1024, 512)), ((), (2560, 2560, 64))]
    bench_cases = [((), (16384, 16384, 16384)), ((), (16384, 32768, 16384)), ((), (32768, 16384, 16384)), ((), (32768, 32768, 16384))]

    def test_args(self, case: tuple) -> tuple[torch.Tensor, ...]:
        M, N, K = case
        return (
            torch.randn(M, K, dtype=torch.bfloat16, device="cuda"),
            torch.randn(K, N, dtype=torch.bfloat16, device="cuda"),
        )

    def bench_flops(self, case: tuple) -> float:
        M, N, K = case
        return 2.0 * M * N * K

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.TILE_M, self.TILE_K), tma_types=[self.A_TMA]),
            TensorSpec(dtype=DType.bf16, granularity=(self.TILE_K, self.TILE_N), tma_types=[self.B_TMA]),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.TILE_M, self.TILE_N), tma_types=[self.D_TMA]),
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

    def block_indices(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> List[Tuple[int, ...]]:
        M, K = src_metas[0].shape
        N = src_metas[1].shape[1]
        m_tiles = M // self.TILE_M
        n_tiles = N // self.TILE_N
        indices = []
        for tile_row, tile_col in self._swizzled_tile_order(m_tiles, n_tiles, self.SUPERGROUP_SIZE):
            indices.append((tile_row, tile_col))
            indices.append((tile_row, tile_col))  # duplicate for CTA 1
        return indices

    def num_instructions(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> int:
        M, K = src_metas[0].shape
        N = src_metas[1].shape[1]
        return (M // self.TILE_M) * (N // self.TILE_N) * 2  # x2 for cluster

    def validate(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> None:
        super().validate(src_metas, dst_metas)
        A_shape = src_metas[0].shape
        B_shape = src_metas[1].shape
        C_shape = dst_metas[0].shape
        if A_shape[1] != B_shape[0]:
            raise RuntimeError(
                f"[MegaKittens] Gemm K-dim mismatch: A has K={A_shape[1]}, B has K={B_shape[0]}"
            )
        if A_shape[1] % self.TILE_K != 0:
            raise RuntimeError(
                f"[MegaKittens] Gemm requires K ({A_shape[1]}) to be a multiple of {self.TILE_K}"
            )
        if C_shape != (A_shape[0], B_shape[1]):
            raise RuntimeError(
                f"[MegaKittens] Gemm output shape mismatch: expected ({A_shape[0]}, {B_shape[1]}), got {C_shape}"
            )
