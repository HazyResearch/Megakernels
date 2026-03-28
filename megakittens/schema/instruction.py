from typing import ClassVar, Tuple

from pydantic import BaseModel, Field, conint

from .itype import IType


class InstructionMeta(BaseModel):
    """Metadata for a unique (itype, src_tensors, dst_tensors) combination assigned an icode."""
    model_config = {"arbitrary_types_allowed": True}
    icode: int
    itype: IType
    src_tensors: Tuple[int, ...]  # TODO: eventually, we want these to be set during runtime
    dst_tensors: Tuple[int, ...]


class Instruction(BaseModel):
    """
    One instruction is consumed by one SM at a time.
    Must match `struct instruction_t` in csrc/schema.cuh.
      - icode: 4B
      - src_tensors: 16B (16-array of uint8)
      - dst_tensors: 8B (8-array of uint8)
      - indices: 64B (16-array of int32)
      - src_barriers: 64B (16-array of uint32)
      - src_barrier_targets: 64B (16-array of int32)
      - num_input_barriers: 1B (uint8)
      - num_reuse_barriers: 1B (uint8)
      - num_dst_barriers: 1B (uint8)
      - _pad: 1B
      - dst_barriers: 32B (8-array of uint32)
    """
    MAX_SRC_TENSORS: ClassVar[int] = 16
    MAX_DST_TENSORS: ClassVar[int] = 8
    MAX_INDICES: ClassVar[int] = 16
    MAX_SRC_BARRIERS: ClassVar[int] = 16
    MAX_SRC_BARRIER_TARGETS: ClassVar[int] = 16
    MAX_DST_BARRIERS: ClassVar[int] = 8

    icode: int
    src_tensors: Tuple[conint(ge=0, le=255), ...] = Field(..., max_length=MAX_SRC_TENSORS)
    dst_tensors: Tuple[conint(ge=0, le=255), ...] = Field(..., max_length=MAX_DST_TENSORS)
    indices: Tuple[conint(ge=-(2**31), le=2**31 - 1), ...] = Field(..., max_length=MAX_INDICES)
    src_barriers: Tuple[conint(ge=0, le=2**32-1), ...] = Field(..., max_length=MAX_SRC_BARRIERS)
    src_barrier_targets: Tuple[conint(ge=-(2**31), le=2**31 - 1), ...] = Field(..., max_length=MAX_SRC_BARRIER_TARGETS)
    num_input_barriers: conint(ge=0, le=255)
    num_reuse_barriers: conint(ge=0, le=255)
    num_dst_barriers: conint(ge=0, le=255)
    dst_barriers: Tuple[conint(ge=0, le=2**32-1), ...] = Field(..., max_length=MAX_DST_BARRIERS)
