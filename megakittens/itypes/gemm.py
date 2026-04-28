import operator
from typing import List, Tuple

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

    A_TMA = st(dtype=DType.bf16, rows=128, cols=64)     # st_bf<Mb/2, Kb>
    A_TMA_T = st(dtype=DType.bf16, rows=64, cols=128)   # st_bf<Kb, Mb/2>
    B_TMA = st(dtype=DType.bf16, rows=64, cols=128)     # st_bf<Kb, Nb/2> (B is K×N)
    B_TMA_T = st(dtype=DType.bf16, rows=128, cols=64)   # st_bf<Nb/2, Kb>
    D_TMA = st(dtype=DType.bf16, rows=128, cols=32)     # st_bf<Mb/2, Nb/EPI_PIPE_DEPTH>

    test_cases = [
        ((), (512, 256, 64)), ((), (512, 256, 256)), ((), (512, 512, 256)),
        ((), (1024, 1024, 512)), ((), (2560, 2560, 64)),
        ((), (2, 512, 256, 64)), ((), (2, 3, 512, 256, 64)),
    ]
    bench_cases = [((), (16384, 16384, 16384)), ((), (16384, 32768, 16384)), ((), (32768, 16384, 16384)), ((), (32768, 32768, 16384))]

    def __init__(self, transpose_a: bool = False, transpose_b: bool = False):
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

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
    def cpp_template(self) -> str:
        ta = "true" if self.transpose_a else "false"
        tb = "true" if self.transpose_b else "false"
        return f"Gemm<MKConfig, MKGlobals, {{tensors}}, {ta}, {tb}>"

    @property
    def inputs(self) -> list[TensorSpec]:
        if self.transpose_a:
            a_gran = (self.TILE_K, self.TILE_M)
            a_tma = self.A_TMA_T
        else:
            a_gran = (self.TILE_M, self.TILE_K)
            a_tma = self.A_TMA
        if self.transpose_b:
            b_gran = (self.TILE_N, self.TILE_K)
            b_tma = self.B_TMA_T
        else:
            b_gran = (self.TILE_K, self.TILE_N)
            b_tma = self.B_TMA
        return [
            TensorSpec(dtype=DType.bf16, granularity=a_gran, tma_types=[a_tma]),
            TensorSpec(dtype=DType.bf16, granularity=b_gran, tma_types=[b_tma]),
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

    def get_gemm_dims(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> Tuple[int, int, int]:
        A_shape = src_metas[0].shape
        B_shape = src_metas[1].shape
        if self.transpose_a:
            K, M = A_shape[-2], A_shape[-1]
        else:
            M, K = A_shape[-2], A_shape[-1]
        if self.transpose_b:
            N = B_shape[-2]
        else:
            N = B_shape[-1]
        return M, K, N

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
                for r, c in self._swizzled_tile_order(D_range[2].size // self.TILE_M, D_range[3].size // self.TILE_N, self.SUPERGROUP_SIZE):
                    if self.transpose_a:
                        a_tile = A_range[3].start // self.TILE_M + r
                    else:
                        a_tile = A_range[2].start // self.TILE_M + r

                    if self.transpose_b:
                        b_tile = B_range[2].start // self.TILE_N + c
                    else:
                        b_tile = B_range[3].start // self.TILE_N + c

                    index = (
                        A_range[0].start + b, A_range[1].start + d, a_tile,
                        B_range[0].start + b, B_range[1].start + d, b_tile,
                        D_range[0].start + b, D_range[1].start + d, D_range[2].start // self.TILE_M + r, D_range[3].start // self.TILE_N + c,
                    )
                    indices.append(index)
                    indices.append(index)  # duplicate for CTA 1
        return indices

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> int:
        D_range = dst_ranges[0]
        return D_range[0].size * D_range[1].size * (D_range[2].size // self.TILE_M) * (D_range[3].size // self.TILE_N) * 2

    def access_regions(
        self,
        block_index: Tuple[int, ...],
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ) -> tuple[list[list[tuple[tuple[int, int], ...]]], list[list[tuple[tuple[int, int], ...]]]]:
        b_A, d_A, a_tile, b_B, d_B, b_tile, b_D, d_D, r_D, c_D = block_index
        M, K, N = self.get_gemm_dims(src_metas, dst_metas)
        if self.transpose_a:
            a_region = ((b_A, b_A + 1), (d_A, d_A + 1), (0, K), (a_tile * self.TILE_M, (a_tile + 1) * self.TILE_M))
        else:
            a_region = ((b_A, b_A + 1), (d_A, d_A + 1), (a_tile * self.TILE_M, (a_tile + 1) * self.TILE_M), (0, K))
        if self.transpose_b:
            b_region = ((b_B, b_B + 1), (d_B, d_B + 1), (b_tile * self.TILE_N, (b_tile + 1) * self.TILE_N), (0, K))
        else:
            b_region = ((b_B, b_B + 1), (d_B, d_B + 1), (0, K), (b_tile * self.TILE_N, (b_tile + 1) * self.TILE_N))
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
        M, K, N = self.get_gemm_dims(src_metas, dst_metas)

        if self.transpose_a:
            if A_range[2].start != 0 or A_range[2].stop != K:
                raise RuntimeError(
                    f"[MegaKittens] Gemm requires full-K range on A (transpose_a=True), got "
                    f"[{A_range[2].start}, {A_range[2].stop}) against K={K}"
                )
        else:
            if A_range[3].start != 0 or A_range[3].stop != K:
                raise RuntimeError(
                    f"[MegaKittens] Gemm requires full-K range on A, got "
                    f"[{A_range[3].start}, {A_range[3].stop}) against K={K}"
                )

        if self.transpose_b:
            if B_range[3].start != 0 or B_range[3].stop != K:
                raise RuntimeError(
                    f"[MegaKittens] Gemm requires full-K range on B (transpose_b=True), got "
                    f"[{B_range[3].start}, {B_range[3].stop}) against K={K}"
                )
        else:
            if B_range[2].start != 0 or B_range[2].stop != K:
                raise RuntimeError(
                    f"[MegaKittens] Gemm requires full-K range on B, got "
                    f"[{B_range[2].start}, {B_range[2].stop}) against K={K}"
                )

        D_effective_shape = D_range.effective_shape
        if D_effective_shape[-2] != M:
            raise RuntimeError(f"[MegaKittens] Gemm effective M mismatch: D={D_effective_shape[-2]} vs A-derived M={M}")
        if D_effective_shape[-1] != N:
            raise RuntimeError(f"[MegaKittens] Gemm effective N mismatch: D={D_effective_shape[-1]} vs B-derived N={N}")
