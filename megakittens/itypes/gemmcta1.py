from typing import List, Tuple

import torch

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorRange, TensorSpec
from ..jit.pykittens import st


@torch.library.custom_op("megakittens::gemmcta1", mutates_args=())
def gemmcta1_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


@gemmcta1_op.register_fake
def _gemmcta1_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty(*a.shape[:-1], b.shape[-1], dtype=a.dtype, device=a.device)


class Gemmcta1(IType):
    TILE_M = 256
    TILE_N = 256
    TILE_K = 64
    SUPERGROUP_SIZE = 8

    cluster_size = 1
    torch_methods_map = {"gemmcta1": None}

    A_TMA = st(dtype=DType.bf16, rows=128, cols=64)  # st_bf<Mb/2, Kb>
    B_TMA = st(dtype=DType.bf16, rows=64, cols=256)  # st_bf<Kb, Nb> (full Nb, 1-CTA)
    D_TMA = st(dtype=DType.bf16, rows=128, cols=32)  # st_bf<Mb/2, Nb/EPI_PIPE_DEPTH>

    test_cases = [
        ((), (256, 256, 64)), ((), (256, 256, 256)), ((), (512, 256, 256)),
        ((), (1024, 1024, 512)), ((), (2560, 2560, 64)),
        ((), (2, 256, 256, 64)), ((), (2, 3, 256, 256, 64)),
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
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        A_range = src_ranges[0]
        B_range = src_ranges[1]
        D_range = dst_ranges[0]
        indices = []
        for b in range(D_range[0].size):
            for d in range(D_range[1].size):
                for r, c in self._swizzled_tile_order(
                    D_range[2].size // self.TILE_M, D_range[3].size // self.TILE_N, self.SUPERGROUP_SIZE
                ):
                    index = (
                        A_range[0].start + b, A_range[1].start + d, A_range[2].start // self.TILE_M + r,
                        B_range[0].start + b, B_range[1].start + d, B_range[3].start // self.TILE_N + c,
                        D_range[0].start + b, D_range[1].start + d, D_range[2].start // self.TILE_M + r, D_range[3].start // self.TILE_N + c,
                    )
                    indices.append(index)
        return indices

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> int:
        D_range = dst_ranges[0]
        return D_range[0].size * D_range[1].size * (D_range[2].size // self.TILE_M) * (D_range[3].size // self.TILE_N)

    def access_regions(
        self,
        block_index: Tuple[int, ...],
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ) -> tuple[list[list[tuple[tuple[int, int], ...]]], list[list[tuple[tuple[int, int], ...]]]]:
        b_A, d_A, r_A, b_B, d_B, c_B, b_D, d_D, r_D, c_D = block_index
        K = src_metas[0].shape[-1]
        a_region = ((b_A, b_A + 1), (d_A, d_A + 1), (r_A * self.TILE_M, (r_A + 1) * self.TILE_M), (0, K))
        b_region = ((b_B, b_B + 1), (d_B, d_B + 1), (0, K), (c_B * self.TILE_N, (c_B + 1) * self.TILE_N))
        d_region = ((b_D, b_D + 1), (d_D, d_D + 1), (r_D * self.TILE_M, (r_D + 1) * self.TILE_M), (c_D * self.TILE_N, (c_D + 1) * self.TILE_N))
        return [[a_region], [b_region]], [[d_region]]

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        A_range = src_ranges[0]
        B_range = src_ranges[1]
        D_range = dst_ranges[0]
        K = src_metas[0].shape[-1]
        if A_range[3].start != 0 or A_range[3].stop != K:
            raise RuntimeError(
                f"[MegaKittens] Gemmcta1 requires full-K range on A, got "
                f"[{A_range[3].start}, {A_range[3].stop}) against K={K}"
            )
        if B_range[2].start != 0 or B_range[2].stop != K:
            raise RuntimeError(
                f"[MegaKittens] Gemmcta1 requires full-K range on B, got "
                f"[{B_range[2].start}, {B_range[2].stop}) against K={K}"
            )

        A_effective_shape = A_range.effective_shape
        B_effective_shape = B_range.effective_shape
        D_effective_shape = D_range.effective_shape
        if A_effective_shape[:-2] != B_effective_shape[:-2] or A_effective_shape[:-2] != D_effective_shape[:-2]:
            raise RuntimeError(
                f"[MegaKittens] Gemmcta1 outer effective-dim mismatch: "
                f"A={A_effective_shape[:-2]} B={B_effective_shape[:-2]} D={D_effective_shape[:-2]}"
            )
        if A_effective_shape[-2] != D_effective_shape[-2]:
            raise RuntimeError(
                f"[MegaKittens] Gemmcta1 effective M mismatch: A={A_effective_shape[-2]} D={D_effective_shape[-2]}"
            )
        if B_effective_shape[-1] != D_effective_shape[-1]:
            raise RuntimeError(
                f"[MegaKittens] Gemmcta1 effective N mismatch: B={B_effective_shape[-1]} D={D_effective_shape[-1]}"
            )
