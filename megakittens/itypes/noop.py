from typing import List, Tuple

from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorSpec


class Noop(IType):
    @property
    def name(self) -> str:
        return "noop"

    @property
    def cpp_template(self) -> str:
        return "Noop<MKConfig, MKGlobals>"

    @property
    def cpp_include(self) -> str:
        return "itypes/noop.cuh"

    @property
    def op_type(self) -> str:
        return "noop"

    @property
    def inputs(self) -> list[TensorSpec]:
        return []

    @property
    def outputs(self) -> list[TensorSpec]:
        return []

    def block_indices(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> List[Tuple[int, ...]]:
        return [()]

    def validate(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> None:
        super().validate(src_metas, dst_metas)
