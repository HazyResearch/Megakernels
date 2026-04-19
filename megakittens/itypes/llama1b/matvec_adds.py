from typing import List, Tuple

import torch

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorMeta, TensorRange, TensorSpec
from ...jit.pykittens import sv, st


@torch.library.custom_op("megakittens::mat_vec_adds", mutates_args=("residual",))
def matvec_adds_op(
    residual: torch.Tensor,
    x: torch.Tensor,
    down_weights: torch.Tensor,
) -> None:
    residual.add_(down_weights[0] @ x)


@matvec_adds_op.register_fake
def _matvec_adds_fake(residual, x, down_weights) -> None:
    pass


BLOCK_SIZE = 16


def _resolve_mat_vec_adds(args, kwargs):
    # n = per-chunk reduction size; wider x (e.g. down_proj's silu_out) is handled via multiple col_offsets
    residual = args[0].meta['val']
    return MatVecAdds(n=residual.shape[-1]), [1]


class MatVecAdds(IType):

    torch_functions_map = {
        torch.ops.megakittens.mat_vec_adds: _resolve_mat_vec_adds,
        torch.ops.megakittens.mat_vec_adds.default: _resolve_mat_vec_adds,
    }

    test_cases = [
        ((2048,), (2048,)),  # (n,), (out_dim,)
    ]
    test_atol = 1e-2
    test_rtol = 1e-2
    bench_cases = [
        ((2048,), (2048,)),
    ]

    @staticmethod
    def test_fn(residual, x, down_weights):
        torch.ops.megakittens.mat_vec_adds(residual, x, down_weights)
        return residual

    def __init__(self, n=0):
        self._n = n

    @property
    def name(self) -> str:
        return "matvec_adds"

    @property
    def cpp_template(self) -> str:
        return f"MatVecAdds<MKConfig, MKGlobals, {self._n}, {{tensors}}>"

    @property
    def cpp_include(self) -> str:
        return "itypes/llama1b/matvec_adds.cuh"

    @property
    def op_type(self) -> str:
        return "matvec_adds"

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(BLOCK_SIZE,),                    # residual
                       tma_types=[sv(dtype=DType.bf16, length=BLOCK_SIZE)]),
            TensorSpec(dtype=DType.bf16, granularity=(1,)),                            # activations
            TensorSpec(dtype=DType.bf16, granularity=(1, 16, 512),                     # weights
                       tma_types=[st(dtype=DType.bf16, rows=16, cols=512)]),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(BLOCK_SIZE,),                    # residual (store_add)
                       tma_types=[sv(dtype=DType.bf16, length=BLOCK_SIZE)]),
        ]

    @property
    def inplace_mapping(self) -> dict[int, int]:
        return {0: 0}

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> int:
        num_blocks = dst_ranges[0][-1].size // BLOCK_SIZE
        num_chunks = src_ranges[1][-1].size // self._n
        return num_blocks * num_chunks

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        out_range = dst_ranges[0]
        x_range = src_ranges[1]
        layer_idx = src_ranges[2][-3].start
        block_start = out_range[-1].start // BLOCK_SIZE
        block_stop = out_range[-1].stop // BLOCK_SIZE
        num_chunks = x_range[-1].size // self._n
        return [
            (layer_idx, b, b + 1, x_range[-1].start + chunk * self._n)
            for chunk in range(num_chunks)
            for b in range(block_start, block_stop)
        ]

    def test_args(self, case):
        out_dim, = case
        n = self._n
        residual = torch.randn(out_dim, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(n, dtype=torch.bfloat16, device="cuda")
        down_weights = torch.randn(1, out_dim, n, dtype=torch.bfloat16, device="cuda")
        return (residual, x, down_weights)

    def access_regions(self, block_index, src_metas, dst_metas):
        layer_idx, start_block, end_block, col_offset = block_index
        n = self._n
        residual_region = ((start_block * BLOCK_SIZE, end_block * BLOCK_SIZE),)
        x_region = ((col_offset, col_offset + n),)
        w_region = ((layer_idx, layer_idx + 1), (start_block * BLOCK_SIZE, end_block * BLOCK_SIZE), (col_offset, col_offset + n))
        out_region = ((start_block * BLOCK_SIZE, end_block * BLOCK_SIZE),)
        return [[residual_region], [x_region], [w_region]], [[out_region]]

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        n = src_ranges[1].effective_shape[-1]
        if n % self._n != 0:
            raise RuntimeError(
                f"[MegaKittens] {self.name}: expected x last dim multiple of {self._n}, got {n}"
            )
