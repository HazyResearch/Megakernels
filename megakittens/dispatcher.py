from __future__ import annotations

import struct
from typing import Any, Sequence

# TODO: completely remove torch dependency and purely rely on CUDA API
import torch

from .dag import DType, TensorMeta
from .instruction import (
    Instruction,
    MAX_DST_BARRIERS,
    MAX_DST_TENSORS,
    MAX_INDICES,
    MAX_SRC_BARRIER_TARGETS,
    MAX_SRC_BARRIERS,
    MAX_SRC_TENSORS,
)


def _pack_uint8s_to_int32s(values: tuple[int, ...], count: int) -> list[int]:
    """Pack *count* uint8 values into count/4 int32s (little-endian, zero-padded)."""
    padded = list(values) + [0] * max(0, count - len(values))
    result: list[int] = []
    for i in range(0, count, 4):
        chunk = bytes(padded[i : i + 4])
        (val,) = struct.unpack("<i", chunk)
        result.append(val)
    return result


def _pack_instruction(inst: Instruction) -> list[int]:
    """Pack a single Instruction into a flat list of 32 int32 values."""
    inst_packed: list[int] = [0] * 32

    # 0 (0-3B): itype
    inst_packed[0] = inst.itype.value

    # 1-4 (4-19B): src_tensors (16 uint8 -> 4 int32)
    inst_packed[1:5] = _pack_uint8s_to_int32s(inst.src_tensors, MAX_SRC_TENSORS)

    # 5-6 (20-27B): dst_tensors (8 uint8 -> 2 int32)
    inst_packed[5:7] = _pack_uint8s_to_int32s(inst.dst_tensors, MAX_DST_TENSORS)

    # 7-20 (28-83B): indices (14 int32, zero-padded)
    indices = list(inst.indices) + [0] * max(0, MAX_INDICES - len(inst.indices))
    inst_packed[7:21] = indices

    # 21-22 (84-91B): src_barriers (8 uint8 -> 2 int32)
    inst_packed[21:23] = _pack_uint8s_to_int32s(inst.src_barriers, MAX_SRC_BARRIERS)

    # 23-30 (92-123B): src_barrier_targets (8 int32, zero-padded)
    targets = list(inst.src_barrier_targets) + [0] * max(
        0, MAX_SRC_BARRIER_TARGETS - len(inst.src_barrier_targets)
    )
    inst_packed[23:31] = targets

    # 31 (124-127B): dst_barrier (4 uint8 -> 1 int32)
    inst_packed[31:32] = _pack_uint8s_to_int32s(inst.dst_barrier, MAX_DST_BARRIERS)

    return inst_packed


def _pack_instructions(instructions: list[Instruction], *, device: str) -> torch.Tensor:
    """Pack a list of Instruction objects into an (N, 32) int32 tensor."""
    buf = [_pack_instruction(inst) for inst in instructions]
    return torch.tensor(buf, dtype=torch.int32, device=device)


# TODO: completely remove torch dependency and purely rely on CUDA API
_MK_DTYPE_TO_TORCH_DTYPE: dict[DType, torch.dtype] = {
    DType.fp64: torch.float64,
    DType.fp32: torch.float32,
    DType.bf16: torch.bfloat16,
    DType.half: torch.float16,
    DType.fp8e4m3: torch.float8_e4m3fn,
    DType.fp8e5m2: torch.float8_e5m2fnuz,
    DType.fp8e8m0: torch.float8_e8m0fnu,
    DType.fp4e2m1x2: torch.float4_e2m1fn_x2,
    DType.int64: torch.int64,
    DType.int32: torch.int32,
    DType.int16: torch.int16,
    DType.int8: torch.int8,
}


def _validate_tensor_against_meta(
    tensor: torch.Tensor, meta: TensorMeta, label: str,
) -> None:
    torch_dtype = _MK_DTYPE_TO_TORCH_DTYPE.get(meta.dtype)
    if torch_dtype is None:
        raise RuntimeError(
            f"[MegaKittens] {label}: unsupported DType {meta.dtype.value}"
        )
    elif tensor.dtype != torch_dtype:
        raise RuntimeError(
            f"[MegaKittens] {label} dtype mismatch "
            f"(expected {torch_dtype}, got {tensor.dtype})"
        )
    elif tuple(tensor.shape) != tuple(meta.shape):
        raise RuntimeError(
            f"[MegaKittens] {label} shape mismatch "
            f"(expected {tuple(meta.shape)}, got {tuple(tensor.shape)})"
        )
    expected_device = str(meta.device)
    actual_device = str(tensor.device)
    if actual_device != expected_device:
        raise RuntimeError(
            f"[MegaKittens] {label} device mismatch "
            f"(expected {expected_device}, got {actual_device})"
        )


class Dispatcher:
    """
    Runtime dispatcher for a compiled MegaKernel plan.

    Stores pre-allocated workspace tensors plus packed int32 tensors for the
    instruction stream and barriers, ready for direct consumption by the CUDA
    MegaKernel.
    """

    def __init__(
        self,
        tensor_metas: list[TensorMeta],
        instructions: list[Instruction],
        num_barriers: int,
        input_tensor_indices: Sequence[int],
        output_tensor_indices: Sequence[int],
    ) -> None:
        if not tensor_metas:
            raise RuntimeError("[MegaKittens] 'tensor_metas' must not be empty")
        if not isinstance(tensor_metas, list) or not all(isinstance(t, TensorMeta) for t in tensor_metas):
            raise RuntimeError(
                "[MegaKittens] 'tensor_metas' must be a list of TensorMeta"
            )
        if not isinstance(instructions, list) or not all(isinstance(i, Instruction) for i in instructions):
            raise RuntimeError(
                "[MegaKittens] 'instructions' must be a list of Instruction"
            )
        if not isinstance(num_barriers, int) or num_barriers < 0:
            raise RuntimeError(
                f"[MegaKittens] 'num_barriers' must be a non-negative int, got {num_barriers!r}"
            )
        if not all(isinstance(i, int) and i >= 0 for i in input_tensor_indices):
            raise RuntimeError(
                "[MegaKittens] 'input_tensor_indices' must contain only non-negative ints"
            )
        if not all(isinstance(i, int) and i >= 0 for i in output_tensor_indices):
            raise RuntimeError(
                "[MegaKittens] 'output_tensor_indices' must contain only non-negative ints"
            )
        num_tensors = len(tensor_metas)
        for idx in input_tensor_indices:
            if idx >= num_tensors:
                raise RuntimeError(
                    f"[MegaKittens] input_tensor_index {idx} out of range [0, {num_tensors})"
                )
        for idx in output_tensor_indices:
            if idx >= num_tensors:
                raise RuntimeError(
                    f"[MegaKittens] output_tensor_index {idx} out of range [0, {num_tensors})"
                )
        devices = {str(m.device) for m in tensor_metas}
        if len(devices) > 1:
            raise RuntimeError(
                f"[MegaKittens] All tensor_metas must share the same device, got {devices}"
            )

        self.device = str(tensor_metas[0].device) # TODO: handle multi-GPU case
        self.tensor_metas = tensor_metas
        self.tensors: list[torch.Tensor | None] = [None] * len(tensor_metas)
        self._materialized = False
        self.instructions = instructions
        self.instruction_tensor = None
        self.num_barriers = num_barriers
        self.barrier_tensor = None
        self.input_tensor_indices = tuple(input_tensor_indices)
        self._input_indices_set = frozenset(input_tensor_indices) # for quick lookup
        self.output_tensor_indices = tuple(output_tensor_indices)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise RuntimeError("[MegaKittens] Dispatcher does not support keyword arguments")

        if len(args) != len(self.input_tensor_indices):
            raise RuntimeError(
                f"[MegaKittens] Dispatcher input count mismatch: expected {len(self.input_tensor_indices)} "
                f"but got {len(args)}"
            )

        if not self._materialized:
            self._materialize(args)
        else:
            self._materialize_inputs(args)

        self._launch()

        outputs = tuple(self.tensors[idx] for idx in self.output_tensor_indices)
        return outputs[0] if len(outputs) == 1 else outputs

    def _materialize(self, args: tuple[Any, ...]) -> None:
        """For the first call: validate & assign inputs, allocate non-input tensors."""
        # Assign input tensor references
        self._materialize_inputs(args)

        # Allocate non-input tensors
        for slot_idx, meta in enumerate(self.tensor_metas):
            if slot_idx in self._input_indices_set:
                continue
            torch_dtype = _MK_DTYPE_TO_TORCH_DTYPE.get(meta.dtype)
            if torch_dtype is None:
                raise RuntimeError(
                    f"[MegaKittens] Tensor slot {slot_idx}: unsupported DType {meta.dtype.value}"
                )
            self.tensors[slot_idx] = torch.empty(
                meta.shape, dtype=torch_dtype, device=str(meta.device),
            )

        # Allocate instruction and barrier tensors
        self.instruction_tensor = _pack_instructions(self.instructions, device=self.device)
        self.barrier_tensor = torch.zeros(
            self.num_barriers, dtype=torch.int32, device=self.device,
        )

        self._materialized = True

    def _materialize_inputs(self, args: tuple[Any, ...]) -> None:
        """Validate & assign input references only."""
        for input_arg_idx, tensor_idx in enumerate(self.input_tensor_indices):
            src = args[input_arg_idx]
            if not isinstance(src, torch.Tensor):
                raise RuntimeError(
                    f"[MegaKittens] Input {input_arg_idx} is not a torch.Tensor "
                    f"(type={type(src).__name__})"
                )
            _validate_tensor_against_meta(
                src, self.tensor_metas[tensor_idx], f"Input {input_arg_idx}"
            )
            self.tensors[tensor_idx] = src

    def _launch(self) -> None:
        pass  # TODO: wire up CUDA kernel launch
