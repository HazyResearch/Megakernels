import operator
from typing import List, Tuple

import torch

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorSpec
from ..jit.pykittens import st


@torch.library.custom_op("megakittens::add", mutates_args=())
def add_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b

@add_op.register_fake
def _add_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(a)


class Add(IType):
    TILE_SIZE = 128
    TILES_PER_INST = 3
    TMA = st(dtype=DType.bf16, rows=128, cols=128)

    torch_functions_map = {
        torch.add: None, operator.add: None,
        torch.ops.aten.add: None, torch.ops.aten.add.default: None, torch.ops.aten.add.Tensor: None,
    }
    torch_methods_map = {"add": None}

    test_cases = [((), (128, 256)), ((), (256, 512)), ((), (512, 1024)), ((), (1024, 2048)), ((), (1280, 2048))]
    bench_cases = [((), (4096, 4096)), ((), (131072, 4096)), ((), (4096, 131072)), ((), (16384, 16384)), ((), (131072, 131072))]

    def test_args(self, case: tuple) -> tuple[torch.Tensor, ...]:
        M, N = case
        return (
            torch.randn(M, N, dtype=torch.bfloat16, device="cuda"),
            torch.randn(M, N, dtype=torch.bfloat16, device="cuda"),
        )

    def bench_bytes(self, case: tuple) -> float:
        M, N = case
        return M * N * 2 * 3

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.TILE_SIZE, self.TILE_SIZE), tma_types=[self.TMA]),
            TensorSpec(dtype=DType.bf16, granularity=(self.TILE_SIZE, self.TILE_SIZE), tma_types=[self.TMA]),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.TILE_SIZE, self.TILE_SIZE), tma_types=[self.TMA]),
        ]

    def block_indices(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> List[Tuple[int, ...]]:
        rows = dst_metas[0].shape[0] // self.TILE_SIZE
        cols = dst_metas[0].shape[1] // self.TILE_SIZE
        indices = []
        for row in range(rows):
            for col in range(0, cols, self.TILES_PER_INST):
                n = min(self.TILES_PER_INST, cols - col)
                indices.append((row, col, n))  # up to 3 tiles per inst
        return indices

    def num_instructions(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> int:
        rows = dst_metas[0].shape[0] // self.TILE_SIZE
        cols = dst_metas[0].shape[1] // self.TILE_SIZE
        return rows * ((cols + self.TILES_PER_INST - 1) // self.TILES_PER_INST)

    def validate(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> None:
        super().validate(src_metas, dst_metas)
        if src_metas[0].shape != src_metas[1].shape:
            raise RuntimeError(
                f"[MegaKittens] Add requires same-shape inputs, got {src_metas[0].shape} and {src_metas[1].shape}. "
                f"Broadcasting is not supported."
            )
