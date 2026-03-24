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
      - indices: 56B (14-array of int32)
      - src_barriers: 8B (8-array of uint8)
      - src_barrier_targets: 32B (8-array of int32)
      - dst_barrier: 4B (4-array of uint8)
    """
    MAX_SRC_TENSORS: ClassVar[int] = 16
    MAX_DST_TENSORS: ClassVar[int] = 8
    MAX_INDICES: ClassVar[int] = 14
    MAX_SRC_BARRIERS: ClassVar[int] = 8
    MAX_SRC_BARRIER_TARGETS: ClassVar[int] = 8
    MAX_DST_BARRIERS: ClassVar[int] = 4

    icode: int
    src_tensors: Tuple[conint(ge=0, le=255), ...] = Field(..., min_length=0, max_length=16)
    dst_tensors: Tuple[conint(ge=0, le=255), ...] = Field(..., min_length=0, max_length=8)
    indices: Tuple[conint(ge=-(2**31), le=2**31 - 1), ...] = Field(..., min_length=0, max_length=14)
    src_barriers: Tuple[conint(ge=0, le=255), ...] = Field(..., min_length=0, max_length=8)
    src_barrier_targets: Tuple[conint(ge=-(2**31), le=2**31 - 1), ...] = Field(..., min_length=0, max_length=8)
    dst_barrier: Tuple[conint(ge=0, le=255), ...] = Field(..., min_length=0, max_length=4)
