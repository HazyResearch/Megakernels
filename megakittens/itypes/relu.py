from typing import List, Tuple

import torch

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorSpec
from ..jit.pykittens import st


@torch.library.custom_op("megakittens::relu", mutates_args=())
def relu_op(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)

@relu_op.register_fake
def _relu_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class Relu(IType):
    TILE_SIZE = 128
    TILES_PER_INST = 7
    TMA = st(dtype=DType.bf16, rows=128, cols=128)

    torch_functions = [
        torch.relu,
        torch.ops.aten.relu, torch.ops.aten.relu.default,
        torch.ops.megakittens.relu, torch.ops.megakittens.relu.default,
    ]
    torch_methods = ["relu"]
    torch_modules = [torch.nn.ReLU, torch.nn.ReLU6]

    test_shapes = [(128, 256), (256, 512), (512, 1024), (1024, 2048), (1280, 2048)]
    bench_shapes = [(4096, 4096), (131072, 4096), (4096, 131072), (16384, 16384), (131072, 131072)]

    @staticmethod
    def test_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.ops.megakittens.relu(x)

    def test_args(self, shape: tuple) -> tuple[torch.Tensor, ...]:
        M, N = shape
        return (torch.randn(M, N, dtype=torch.bfloat16, device="cuda"),)

    def bench_bytes(self, shape: tuple) -> float:
        M, N = shape
        return M * N * 2 * 2  # 2 bytes/bf16, 2 tensors (1 read + 1 write)

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
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
                indices.append((row, col, n))
        return indices

    def num_instructions(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> int:
        rows = dst_metas[0].shape[0] // self.TILE_SIZE
        cols = dst_metas[0].shape[1] // self.TILE_SIZE
        return rows * ((cols + self.TILES_PER_INST - 1) // self.TILES_PER_INST)

    def validate(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> None:
        super().validate(src_metas, dst_metas)
        if src_metas[0].shape != dst_metas[0].shape:
            raise RuntimeError(
                f"[MegaKittens] Relu output shape {dst_metas[0].shape} doesn't match input shape {src_metas[0].shape}"
            )
