from abc import ABC, abstractmethod
from collections.abc import Callable
import re
from typing import List, Tuple

import torch

from .tensor import TensorMeta, TensorSpec


class IType(ABC):
    """Instruction type. Inherit with a subclass to define a new instruction type."""

    torch_functions_map: dict[Callable, Callable | None] = {}
    torch_methods_map: dict[str, Callable | None] = {}
    torch_modules_map: dict[type, Callable | None] = {}

    test_cases: list[tuple] = []
    test_atol: float = 0.0
    test_rtol: float = 0.0
    bench_cases: list[tuple] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cls.__name__).lower()
        mk_op = getattr(torch.ops.megakittens, name, None)
        if mk_op is None:
            raise RuntimeError(f"[MegaKittens] {cls.__name__} requires custom op torch.ops.megakittens.{name}")
        cls.torch_functions_map = {mk_op: None, mk_op.default: None, **cls.torch_functions_map}
        if "test_fn" not in cls.__dict__:
            def _make_test_fn(op):
                def test_fn(*args):
                    return op(*args)
                test_fn.__qualname__ = f"{cls.__name__}.test_fn"
                return test_fn
            cls.test_fn = staticmethod(_make_test_fn(mk_op))
        _register_itype(cls)

    @property
    def name(self) -> str:
        """Default: snake_case of class name. Override if different."""
        return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", type(self).__name__).lower()

    @property
    def cpp_template(self) -> str:
        """Default: ``ClassName<MKConfig, MKGlobals, {tensors}>``. Override if different."""
        return f"{type(self).__name__}<MKConfig, MKGlobals, {{tensors}}>"

    @property
    def cpp_include(self) -> str:
        """Default: ``itypes/<name>.cuh``. Override if different."""
        return f"itypes/{self.name}.cuh"

    @property
    @abstractmethod
    def inputs(self) -> list[TensorSpec]:
        """Specs for each input tensor."""
        ...

    @property
    @abstractmethod
    def outputs(self) -> list[TensorSpec]:
        """Specs for each output tensor."""
        ...

    @abstractmethod
    def test_args(self, case: tuple) -> tuple:
        """Create input tensors for a given test/benchmark shape."""
        ...

    @abstractmethod
    def block_indices(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> List[Tuple[int, ...]]:
        """Return instruction coordinate tuples for one node. Each becomes one instruction's indices."""
        ...

    def num_instructions(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> int:
        """Number of instructions this node generates. Override if computable without building the full list."""
        return len(self.block_indices(src_metas, dst_metas))

    def validate(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> None:
        """Validate input/output TensorMeta against specs. Override for custom checks."""
        if len(src_metas) != len(self.inputs):
            raise RuntimeError(
                f"[MegaKittens] {self.name} requires {len(self.inputs)} inputs, got {len(src_metas)}"
            )
        if len(dst_metas) != len(self.outputs):
            raise RuntimeError(
                f"[MegaKittens] {self.name} requires {len(self.outputs)} outputs, got {len(dst_metas)}"
            )
        for label, metas, specs in [("input", src_metas, self.inputs), ("output", dst_metas, self.outputs)]:
            for i, (meta, spec) in enumerate(zip(metas, specs)):
                if meta.dtype != spec.dtype:
                    raise RuntimeError(
                        f"[MegaKittens] {self.name} {label} {i}: expected dtype {spec.dtype.value}, got {meta.dtype.value}"
                    )
                for dim, gran in enumerate(spec.granularity):
                    if meta.shape[dim] % gran != 0:
                        raise RuntimeError(
                            f"[MegaKittens] {self.name} {label} {i} dim {dim}: {meta.shape[dim]} not a multiple of {gran}"
                        )

    _itype_cache: dict[str, "IType"] = {}
    @classmethod
    def from_optype(cls, op_type: str) -> "IType":
        if not cls._itype_cache:
            for subclass in cls.__subclasses__():  # TODO: in the future, there will exist multiple itypes per optype
                itype = subclass()
                cls._itype_cache[itype.op_type] = itype
        if op_type not in cls._itype_cache:
            raise ValueError(f"[MegaKittens] No IType for OpType '{op_type}'")
        return cls._itype_cache[op_type]

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))
