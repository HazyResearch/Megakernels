from typing import List, Tuple

import torch

from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorSpec


@torch.library.custom_op("megakittens::noop", mutates_args=())
def noop_op() -> None:
    pass

@noop_op.register_fake
def _noop_fake() -> None:
    pass


class Noop(IType):
    torch_functions = []
    test_cases: list[tuple] = []
    bench_cases: list[tuple] = []

    def test_args(self, shape):
        return ()

    @property
    def cpp_template(self) -> str:
        return "Noop<MKConfig, MKGlobals>"

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
