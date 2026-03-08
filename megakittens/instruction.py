from enum import Enum
from typing import Tuple

from pydantic import BaseModel, Field, conint


MAX_SRC_TENSORS = 16
MAX_DST_TENSORS = 8
MAX_INDICES = 14
MAX_SRC_BARRIERS = 8
MAX_SRC_BARRIER_TARGETS = 8
MAX_DST_BARRIERS = 4

_uint = conint(ge=0, le=255)
_int32 = conint(ge=-(2**31), le=2**31 - 1)


class IType(Enum):
    relu = 0
    mm_bf16_bf16_fp32_bf16 = 1
    add = 2


class Instruction(BaseModel):
    """
    One instruction is consumed by one SM at a time.
    One instruction is 128B, composed as following:
      - itype: 4B
      - src_tensors: 16B (16-array of uint8)
      - dst_tensors: 8B (8-array of uint8)
      - indices: 56B (14-array of int32)
      - src_barriers: 8B (8-array of uint8)
      - src_barrier_targets: 32B (8-array of int32)
    - dst_barrier: 4B (4-array of uint8)
    """
    itype: IType
    src_tensors: Tuple[_uint, ...] = Field(..., min_items=0, max_items=MAX_SRC_TENSORS)
    dst_tensors: Tuple[_uint, ...] = Field(..., min_items=0, max_items=MAX_DST_TENSORS)
    indices: Tuple[_int32, ...] = Field(..., min_items=0, max_items=MAX_INDICES)
    src_barriers: Tuple[_uint, ...] = Field(..., min_items=0, max_items=MAX_SRC_BARRIERS)
    src_barrier_targets: Tuple[_int32, ...] = Field(..., min_items=0, max_items=MAX_SRC_BARRIER_TARGETS)
    dst_barrier: Tuple[_uint, ...] = Field(..., min_items=0, max_items=MAX_DST_BARRIERS)
