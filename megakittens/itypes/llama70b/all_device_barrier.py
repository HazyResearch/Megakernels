"""AllDeviceBarrier — LLaMA-70B cross-device global sync (reference opcode 13).

Mirrors ``csrc/itypes/reference/all_device_barrier.cu``: every SM on every
device bumps a SYS-scope counter and spin-waits until it reaches
``gridDim.x * NUM_DEVICES``. Used between layers or between prefill/decode
phases to establish a cross-device happens-before.

The IType has no inputs and one PGL int32 output (shape ``(num_slots,)``).
Each instruction consumes one slot (``indices[0]``); the framework allocates
the output as zero-initialized per-device tensors. After a launch, device
0's slot should hold ``gridDim.x * NUM_DEVICES`` — that's the test hook.
"""

from typing import List, Tuple

import torch

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorMeta, TensorRange, TensorSpec


_PGL_NUM_DEVICES_DEFAULT = 8


@torch.library.custom_op("megakittens::all_device_barrier", mutates_args=())
def all_device_barrier_op(bar: torch.Tensor) -> torch.Tensor:
    # Eager fallback: return a zero tensor the same shape as `bar`. The real
    # sync lives in the kernel; eager just makes the op registration happy.
    return torch.zeros_like(bar)


@all_device_barrier_op.register_fake
def _all_device_barrier_fake(bar: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(bar)


class AllDeviceBarrier(IType):
    """Cross-device SYS-scope barrier (reference opcode 13)."""

    NUM_DEVICES = _PGL_NUM_DEVICES_DEFAULT

    torch_functions_map: dict = {}
    test_cases: list[tuple] = []
    bench_cases: list[tuple] = []

    @property
    def cpp_template(self) -> str:
        return "AllDeviceBarrier<MKConfig, MKGlobals, {tensors}>"

    @property
    def cpp_include(self) -> str:
        return "itypes/llama70b/all_device_barrier.cuh"

    def test_args(self, case: tuple):
        raise NotImplementedError(
            "AllDeviceBarrier is a cross-device sync primitive; exercise it "
            "inside a MultiDispatcher DAG (tests/test_all_device_barrier.py), "
            "not a single-device unit test."
        )

    @property
    def inputs(self) -> list[TensorSpec]:
        # Dummy handle — the scheduler requires at least one input tensor
        # per DAG. The kernel never reads it; it only writes the output.
        return [
            TensorSpec(
                dtype=DType.int32,
                granularity=(1,),
                tma_types=[],
                num_devices=self.NUM_DEVICES,
            ),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                dtype=DType.int32,
                granularity=(1,),
                tma_types=[],
                num_devices=self.NUM_DEVICES,
            ),
        ]

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        num_slots = dst_metas[0].shape[0]
        return [(slot,) for slot in range(num_slots)]

    def access_regions(
        self,
        block_index: Tuple[int, ...],
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ) -> tuple[list[list[tuple[tuple[int, int], ...]]], list[list[tuple[tuple[int, int], ...]]]]:
        (slot_idx,) = block_index
        region = [((slot_idx, slot_idx + 1),)]
        return [region], [region]

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        for label, metas in (("input", src_metas), ("output", dst_metas)):
            if len(metas[0].shape) != 1:
                raise RuntimeError(
                    f"[MegaKittens] AllDeviceBarrier expects 1D int32 {label}, "
                    f"got shape {metas[0].shape}"
                )
