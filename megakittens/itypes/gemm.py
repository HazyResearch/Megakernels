import operator
from typing import List, Optional, Tuple

import torch

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorRange, TensorSpec
from ..jit.pykittens import st


@torch.library.custom_op("megakittens::gemm", mutates_args=())
def gemm_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


@gemm_op.register_fake
def _gemm_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty(*a.shape[:-1], b.shape[-1], dtype=a.dtype, device=a.device)


class Gemm(IType):
    TILE_M = 512
    TILE_N = 256
    TILE_K = 64
    SUPERGROUP_SIZE = 8

    torch_functions_map = {
        torch.matmul: None, torch.mm: None, torch.bmm: None, operator.matmul: None,
        torch.ops.aten.mm: None, torch.ops.aten.mm.default: None,
        torch.ops.aten.bmm: None, torch.ops.aten.bmm.default: None,
        torch.ops.aten.matmul: None, torch.ops.aten.matmul.default: None,
    }
    torch_methods_map = {"gemm": None}

    A_TMA = st(dtype=DType.bf16, rows=128, cols=64) # st_bf<Mb/2, Kb>
    B_TMA = st(dtype=DType.bf16, rows=64, cols=128) # st_bf<Kb, Nb/2> (B is K×N)
    D_TMA = st(dtype=DType.bf16, rows=128, cols=32) # st_bf<Mb/2, Nb/EPI_PIPE_DEPTH>

    test_cases = [
        ((), (512, 256, 64)), ((), (512, 256, 256)), ((), (512, 512, 256)),
        ((), (1024, 1024, 512)), ((), (2560, 2560, 64)),
        ((), (2, 512, 256, 64)), ((), (2, 3, 512, 256, 64)),
    ]
    bench_cases = [((), (16384, 16384, 16384)), ((), (16384, 32768, 16384)), ((), (32768, 16384, 16384)), ((), (32768, 32768, 16384))]

    def test_args(self, case: tuple) -> tuple[torch.Tensor, ...]:
        *outer, R, C, K = case
        return (
            torch.randn(*outer, R, K, dtype=torch.bfloat16, device="cuda"),
            torch.randn(*outer, K, C, dtype=torch.bfloat16, device="cuda"),
        )

    def bench_flops(self, case: tuple) -> float:
        *outer, R, C, K = case
        n = 1
        for d in outer:
            n *= d
        return 2.0 * n * R * C * K

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

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[Optional[TensorRange], ...] | None = None,
        dst_ranges: Tuple[Optional[TensorRange], ...] | None = None,
    ) -> List[Tuple[int, ...]]:
        if src_ranges is not None or dst_ranges is not None:
            raise RuntimeError("[MegaKittens] Gemm does not yet support tensor ranges")
        B, D, _R, _K = (1,) * (4 - len(src_metas[0].shape)) + src_metas[0].shape
        R = _R // self.TILE_M
        C = src_metas[1].shape[-1] // self.TILE_N
        indices = []
        for b in range(B):
            for d in range(D):
                for r, c in self._swizzled_tile_order(R, C, self.SUPERGROUP_SIZE):
                    indices.append((b, d, r, c))
                    indices.append((b, d, r, c))  # duplicate for CTA 1
        return indices

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[Optional[TensorRange], ...] | None = None,
        dst_ranges: Tuple[Optional[TensorRange], ...] | None = None,
    ) -> int:
        if src_ranges is not None or dst_ranges is not None:
            raise RuntimeError("[MegaKittens] Gemm does not yet support tensor ranges")
        B, D, _R, _K = (1,) * (4 - len(src_metas[0].shape)) + src_metas[0].shape
        R = _R // self.TILE_M
        C = src_metas[1].shape[-1] // self.TILE_N
        return B * D * R * C * 2

    def access_regions(self, block_index, src_metas, dst_metas):
        b, d, r, c = block_index
        K = src_metas[0].shape[-1]
        a_region = ((b, b + 1), (d, d + 1), (r * self.TILE_M, (r + 1) * self.TILE_M), (0, K))
        b_region = ((b, b + 1), (d, d + 1), (0, K), (c * self.TILE_N, (c + 1) * self.TILE_N))
        d_region = ((b, b + 1), (d, d + 1), (r * self.TILE_M, (r + 1) * self.TILE_M), (c * self.TILE_N, (c + 1) * self.TILE_N))
        return [a_region, b_region], [d_region]

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[Optional[TensorRange], ...] | None = None,
        dst_ranges: Tuple[Optional[TensorRange], ...] | None = None,
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        if src_ranges is not None or dst_ranges is not None:
            raise RuntimeError("[MegaKittens] Gemm does not yet support tensor ranges")
        A_shape = src_metas[0].shape
        B_shape = src_metas[1].shape
        C_shape = dst_metas[0].shape
        if A_shape[:-2] != B_shape[:-2]:
            raise RuntimeError(
                f"[MegaKittens] Gemm outer dims mismatch: A has {A_shape[:-2]}, B has {B_shape[:-2]}"
            )
        if A_shape[-1] != B_shape[-2]:
            raise RuntimeError(
                f"[MegaKittens] Gemm K-dim mismatch: A has K={A_shape[-1]}, B has K={B_shape[-2]}"
            )
        if A_shape[-1] % self.TILE_K != 0:
            raise RuntimeError(
                f"[MegaKittens] Gemm requires K ({A_shape[-1]}) to be a multiple of {self.TILE_K}"
            )
        if C_shape[-2:] != (A_shape[-2], B_shape[-1]):
            raise RuntimeError(
                f"[MegaKittens] Gemm output shape mismatch: expected ({A_shape[-2]}, {B_shape[-1]}), got {C_shape[-2:]}"
            )
