from typing import List, Tuple

import torch

from ..dispatcher import Dispatcher
from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorRange, TensorSpec
from ..jit.pykittens import st


@torch.library.custom_op("megakittens::copy", mutates_args=())
def copy_op(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x.to(dtype)


@copy_op.register_fake
def _copy_fake(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty_like(x, dtype=dtype)


def _resolve_from_custom_op(args, kwargs):
    x_node = args[0]
    dst_torch_dtype = args[1] if len(args) > 1 else kwargs["dtype"]
    src_dtype = DType.from_torch(x_node.meta['val'].dtype)
    dst_dtype = DType.from_torch(dst_torch_dtype)
    return Copy(src_dtype=src_dtype, dst_dtype=dst_dtype)


def _resolve_to_copy(args, kwargs):
    src_node = args[0]
    src_torch_dtype = src_node.meta['val'].dtype
    dst_torch_dtype = kwargs.get('dtype', src_torch_dtype)
    return Copy(src_dtype=DType.from_torch(src_torch_dtype), dst_dtype=DType.from_torch(dst_torch_dtype))


class Copy(IType):
    TILE_ROWS = 128
    MAX_TILES_PER_INST = 2
    # tile_cols = PAGE_SIZE / (TILE_ROWS * max(src_dtype_size, dst_dtype_size))

    torch_functions_map = {
        torch.ops.megakittens.copy: _resolve_from_custom_op,
        torch.ops.megakittens.copy.default: _resolve_from_custom_op,
        torch.ops.aten._to_copy: _resolve_to_copy,
        torch.ops.aten._to_copy.default: _resolve_to_copy,
    }

    test_cases = [
        ((DType.bf16, DType.fp32), shape)
        for shape in [(128, 64), (512, 256), (2, 128, 128), (2, 3, 128, 256)]
    ] + [
        ((DType.fp32, DType.bf16), shape)
        for shape in [(128, 64), (512, 256), (2, 128, 128), (2, 3, 128, 256)]
    ]
    test_atol = 0.0
    test_rtol = 0.0

    def __init__(self, src_dtype: DType = DType.bf16, dst_dtype: DType = DType.fp32):
        self.src_dtype = src_dtype
        self.dst_dtype = dst_dtype

    @property
    def tile_cols(self) -> int:
        return Dispatcher.PAGE_SIZE // (self.TILE_ROWS * max(self.src_dtype.size, self.dst_dtype.size))

    @property
    def src_tma(self) -> st:
        return st(dtype=self.src_dtype, rows=self.TILE_ROWS, cols=self.tile_cols)

    @property
    def dst_tma(self) -> st:
        return st(dtype=self.dst_dtype, rows=self.TILE_ROWS, cols=self.tile_cols)

    def test_args(self, case: tuple) -> tuple:
        x = torch.randn(*case, dtype=self.src_dtype.torch_dtype, device="cuda")
        return (x, self.dst_dtype.torch_dtype)

    @property
    def cpp_template(self) -> str:
        return f"Copy<MKConfig, MKGlobals, {self.src_dtype.cpp_dtype}, {self.dst_dtype.cpp_dtype}, {{tensors}}>"

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=self.src_dtype, granularity=(self.TILE_ROWS, self.tile_cols), tma_types=[self.src_tma]),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=self.dst_dtype, granularity=(self.TILE_ROWS, self.tile_cols), tma_types=[self.dst_tma]),
        ]

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        src_range = src_ranges[0]
        dst_range = dst_ranges[0]
        tile_cols = self.tile_cols
        indices = []
        for b in range(dst_range[0].size):
            for d in range(dst_range[1].size):
                for r in range(dst_range[2].size // self.TILE_ROWS):
                    for c in range(0, dst_range[3].size // tile_cols, self.MAX_TILES_PER_INST):
                        n = min(self.MAX_TILES_PER_INST, dst_range[3].size // tile_cols - c)
                        indices.append((
                            src_range[0].start + b, src_range[1].start + d, src_range[2].start // self.TILE_ROWS + r, src_range[3].start // tile_cols + c,
                            dst_range[0].start + b, dst_range[1].start + d, dst_range[2].start // self.TILE_ROWS + r, dst_range[3].start // tile_cols + c,
                            n,
                        ))
        return indices

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> int:
        dst_range = dst_ranges[0]
        return (dst_range[0].size * dst_range[1].size
                * (dst_range[2].size // self.TILE_ROWS)
                * ((dst_range[3].size // self.tile_cols + self.MAX_TILES_PER_INST - 1) // self.MAX_TILES_PER_INST))

    def access_regions(
        self,
        block_index: Tuple[int, ...],
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ) -> tuple[list[list[tuple[tuple[int, int], ...]]], list[list[tuple[tuple[int, int], ...]]]]:
        b_src, d_src, r_src, c_src, b_dst, d_dst, r_dst, c_dst, n = block_index
        tile_cols = self.tile_cols
        src_region = ((b_src, b_src + 1), (d_src, d_src + 1), (r_src * self.TILE_ROWS, (r_src + 1) * self.TILE_ROWS), (c_src * tile_cols, (c_src + n) * tile_cols))
        dst_region = ((b_dst, b_dst + 1), (d_dst, d_dst + 1), (r_dst * self.TILE_ROWS, (r_dst + 1) * self.TILE_ROWS), (c_dst * tile_cols, (c_dst + n) * tile_cols))
        return [[src_region]], [[dst_region]]

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        if src_ranges[0].effective_shape != dst_ranges[0].effective_shape:
            raise RuntimeError(
                f"[MegaKittens] Copy effective shape mismatch: "
                f"src={src_ranges[0].effective_shape} dst={dst_ranges[0].effective_shape}"
            )
