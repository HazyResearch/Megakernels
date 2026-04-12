from abc import ABC, abstractmethod
from collections.abc import Callable
import re
from typing import List, Tuple

import torch

from .tensor import TensorMeta, TensorSpec


# Global dispatch maps: target -> IType | list[Callable]
# Resolvers (Callable) return IType or (IType, list[int]) where list[int] is aten_output_indices.
_CALL_FUNCTION_MAP: dict[Callable, "IType | list[Callable]"] = {}
_CALL_METHOD_MAP: dict[str, "IType | list[Callable]"] = {}
_CALL_MODULE_MAP: dict[type, "IType | list[Callable]"] = {}


def _register_itype(cls):
    for global_map, class_map in [
        (_CALL_FUNCTION_MAP, cls.torch_functions_map),
        (_CALL_METHOD_MAP, cls.torch_methods_map),
        (_CALL_MODULE_MAP, cls.torch_modules_map),
    ]:
        for key, value in class_map.items():
            existing = global_map.get(key)
            if callable(value):
                if isinstance(existing, IType):
                    raise RuntimeError(f"[MegaKittens] Conflict: {key!r} already registered as plain IType, cannot add resolver")
                global_map.setdefault(key, []).append(value)
            else:
                if isinstance(existing, list):
                    raise RuntimeError(f"[MegaKittens] Conflict: {key!r} already registered as resolver, cannot add plain IType")
                global_map[key] = cls()


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
        """Default: ``itypes/<name>.cuh`` or ``itypes/<subdir>/<name>.cuh``."""
        name = type(self).__module__.removeprefix("megakittens.itypes.").replace(".", "/")
        return f"itypes/{name}.cuh"

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
                if len(meta.shape) > 4:
                    raise RuntimeError(
                        f"[MegaKittens] {self.name} {label} {i}: tensors must be at most 4D, got {len(meta.shape)}D"
                    )
                if len(meta.shape) < len(spec.granularity):
                    raise RuntimeError(
                        f"[MegaKittens] {self.name} {label} {i}: tensor is {len(meta.shape)}D but granularity requires at least {len(spec.granularity)}D"
                    )
                if meta.dtype != spec.dtype:
                    raise RuntimeError(
                        f"[MegaKittens] {self.name} {label} {i}: expected dtype {spec.dtype.value}, got {meta.dtype.value}"
                    )
                offset = len(meta.shape) - len(spec.granularity)
                for g_dim, gran in enumerate(spec.granularity):
                    if meta.shape[offset + g_dim] % gran != 0:
                        raise RuntimeError(
                            f"[MegaKittens] {self.name} {label} {i} dim {offset + g_dim}: "
                            f"{meta.shape[offset + g_dim]} not a multiple of {gran}"
                        )

    @abstractmethod
    def access_regions(
        self,
        block_index: Tuple[int, ...],
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ) -> tuple[list[tuple[tuple[int, int], ...]], list[tuple[tuple[int, int], ...]]]:
        """Per-instruction tile regions for fine-grained barriers.
        Returns (src_regions, dst_regions); one tuple of (start, end) ranges per tensor."""
        ...

    @classmethod
    def from_torch(cls, target: Callable | str | type, args=(), kwargs={}) -> "IType | tuple[IType, list[int]]":
        if isinstance(target, str):
            mapping = _CALL_METHOD_MAP
            label = "call_method target"
        elif isinstance(target, type):
            mapping = _CALL_MODULE_MAP
            label = "module type"
        elif callable(target):
            mapping = _CALL_FUNCTION_MAP
            label = "call_function target"
        else:
            raise TypeError(f"[MegaKittens] Cannot resolve IType from {type(target).__name__}")

        if target not in mapping:
            raise RuntimeError(f"[MegaKittens] Unsupported {label}: {target!r}")
        entry = mapping[target]

        if isinstance(entry, list):
            for resolver in entry:
                if not callable(resolver):
                    raise RuntimeError(f"[MegaKittens] Resolver for {label} {target!r} is not callable: {resolver!r}")
                result = resolver(args, kwargs)
                if result is None:
                    continue
                if isinstance(result, tuple):
                    if len(result) != 2 or not isinstance(result[0], IType) or not isinstance(result[1], list):
                        raise RuntimeError(
                            f"[MegaKittens] Resolver for {label} {target!r} returned invalid tuple: {result!r}"
                        )
                elif not isinstance(result, IType):
                    raise RuntimeError(
                        f"[MegaKittens] Resolver for {label} {target!r} returned invalid type: {type(result).__name__}"
                    )
                return result
            raise RuntimeError(f"[MegaKittens] No matching resolver for {label}: {target!r}")
        else:
            if not isinstance(entry, IType):
                raise RuntimeError(f"[MegaKittens] Invalid entry for {label} {target!r}: expected IType, got {type(entry).__name__}")
            return entry

    @property
    def _id(self) -> tuple:
        return (type(self), tuple(sorted(self.__dict__.items())))

    def __repr__(self) -> str:
        if self.__dict__:
            fields = ", ".join(f"{k}={v!r}" for k, v in sorted(self.__dict__.items()))
            return f"{type(self).__name__}({fields})"
        return f"{type(self).__name__}()"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IType):
            return NotImplemented
        return self._id == other._id

    def __hash__(self) -> int:
        return hash(self._id)
