import itertools
from typing import List, Tuple

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorSpec
from ..jit.pykittens import st


class Add(IType):
    TILE_SIZE = 128
    TMA = st(dtype=DType.bf16, rows=128, cols=128, swizzle=False)

    @property
    def name(self) -> str:
        return "add"

    @property
    def cpp_template(self) -> str:
        return "Add<MKConfig, MKGlobals, {tensors}>"

    @property
    def cpp_include(self) -> str:
        return "itypes/add.cuh"

    @property
    def op_type(self) -> str:
        return "add"

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
        ranges = [range(dim // self.TILE_SIZE) for dim in dst_metas[0].shape]
        return list(itertools.product(*ranges))

    def num_instructions(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> int:
        result = 1
        for dim in dst_metas[0].shape:
            result *= dim // self.TILE_SIZE
        return result

    def validate(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> None:
        super().validate(src_metas, dst_metas)
        if src_metas[0].shape != src_metas[1].shape:
            raise RuntimeError(
                f"[MegaKittens] Add requires same-shape inputs, got {src_metas[0].shape} and {src_metas[1].shape}. "
                f"Broadcasting is not supported."
            )
