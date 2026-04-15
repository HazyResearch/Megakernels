from typing import List, Optional, Tuple

import torch

from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorRange, TensorSpec


@torch.library.custom_op("megakittens::noop", mutates_args=())
def noop_op() -> None:
    pass


@noop_op.register_fake
def _noop_fake() -> None:
    pass


class Noop(IType):
    torch_functions_map: dict = {}
    test_cases: list[tuple] = []
    bench_cases: list[tuple] = []

    def test_args(self, case):
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

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[Optional[TensorRange], ...] | None = None,
        dst_ranges: Tuple[Optional[TensorRange], ...] | None = None,
    ) -> List[Tuple[int, ...]]:
        if src_ranges is not None or dst_ranges is not None:
            raise RuntimeError("[MegaKittens] Noop does not yet support tensor ranges")
        return [()]

    def access_regions(self, block_index, src_metas, dst_metas):
        return [], []

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[Optional[TensorRange], ...] | None = None,
        dst_ranges: Tuple[Optional[TensorRange], ...] | None = None,
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        if src_ranges is not None or dst_ranges is not None:
            raise RuntimeError("[MegaKittens] Noop does not yet support tensor ranges")
