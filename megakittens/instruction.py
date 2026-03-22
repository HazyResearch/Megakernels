from abc import ABC, abstractmethod
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


class IType(ABC):
    """Instruction type. Inherit with a subclass to define a new op."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def tile_size(self) -> int:
        """Tile shape per SM per instruction."""
        ...

    @property
    @abstractmethod
    def cpp_template(self) -> str:
        """
        C++ op template string for JIT codegen.
        Use ``{tensors}`` as placeholder for the comma-separated tensor indices.
        Example: ``"Add<MKConfig, MKGlobals, {tensors}>"``
        """
        ...

    @property
    @abstractmethod
    def cpp_include(self) -> str:
        """Header to include for this op. Example: ``"ops/add.cuh"``"""
        ...

    @property
    @abstractmethod
    def op_type(self) -> str:
        """The OpType value this instruction implements (e.g. ``"add"``, ``"matmul"``)."""
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))


class Add(IType):
    @property
    def name(self) -> str:
        return "add"

    @property
    def tile_size(self) -> int:
        return 4

    @property
    def cpp_template(self) -> str:
        return "Add<MKConfig, MKGlobals, {tensors}>"

    @property
    def cpp_include(self) -> str:
        return "ops/add.cuh"

    @property
    def op_type(self) -> str:
        return "add"


class OpMeta(BaseModel):
    """Metadata for a unique (itype, src_tensors, dst_tensors) combination assigned an opcode."""
    model_config = {"arbitrary_types_allowed": True}
    opcode: int
    itype: IType
    src_tensors: Tuple[int, ...]  # TODO: eventually, we want these to be set during runtime
    dst_tensors: Tuple[int, ...]


class Instruction(BaseModel):
    """
    One instruction is consumed by one SM at a time.
    One instruction is 128B, composed as following:
      - opcode: 4B
      - src_tensors: 16B (16-array of uint8)
      - dst_tensors: 8B (8-array of uint8)
      - indices: 56B (14-array of int32)
      - src_barriers: 8B (8-array of uint8)
      - src_barrier_targets: 32B (8-array of int32)
    - dst_barrier: 4B (4-array of uint8)
    """
    opcode: int
    src_tensors: Tuple[_uint, ...] = Field(..., min_items=0, max_items=MAX_SRC_TENSORS)
    dst_tensors: Tuple[_uint, ...] = Field(..., min_items=0, max_items=MAX_DST_TENSORS)
    indices: Tuple[_int32, ...] = Field(..., min_items=0, max_items=MAX_INDICES)
    src_barriers: Tuple[_uint, ...] = Field(..., min_items=0, max_items=MAX_SRC_BARRIERS)
    src_barrier_targets: Tuple[_int32, ...] = Field(..., min_items=0, max_items=MAX_SRC_BARRIER_TARGETS)
    dst_barrier: Tuple[_uint, ...] = Field(..., min_items=0, max_items=MAX_DST_BARRIERS)
