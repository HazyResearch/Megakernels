from typing import List, Tuple

import torch

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorRange, TensorSpec


@torch.library.custom_op("megakittens::pos_id_increment", mutates_args=("pos_id",))
def pos_id_increment_op(pos_id: torch.Tensor) -> None:
    pos_id.add_(1)


@pos_id_increment_op.register_fake
def _pos_id_increment_fake(pos_id) -> None:
    pass


def _resolve_pos_id_increment(args, kwargs):
    return PosIdIncrement(), [1]


class PosIdIncrement(IType):

    torch_functions_map = {
        torch.ops.megakittens.pos_id_increment: _resolve_pos_id_increment,
        torch.ops.megakittens.pos_id_increment.default: _resolve_pos_id_increment,
    }

    test_cases = [
        ((), (0,)),
    ]
    bench_cases = [
        ((), (0,)),
    ]

    @staticmethod
    def test_fn(pos_id):
        torch.ops.megakittens.pos_id_increment(pos_id)
        return pos_id

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.int32, granularity=(1,)),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.int32, granularity=(1,)),
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
        return 1

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        return [(0,)]

    def test_args(self, case):
        pos_id_val, = case
        return (torch.tensor([pos_id_val], dtype=torch.int32, device="cuda"),)

    def access_regions(self, block_index, src_metas, dst_metas):
        return [[((0, 1),)]], [[((0, 1),)]]
