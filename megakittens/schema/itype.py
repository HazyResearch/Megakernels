from abc import ABC, abstractmethod
import re
from typing import List, Tuple

from .optype import register_optype
from .tensor import TensorMeta, TensorSpec


class IType(ABC):
    """Instruction type. Inherit with a subclass to define a new instruction type."""

    torch_functions: list = []      # e.g. [torch.add, operator.add, torch.ops.aten.add, ...]
    torch_methods: list[str] = []   # e.g. ["add"]
    torch_modules: list[type] = []  # e.g. [torch.nn.ReLU]

    test_shapes: list[tuple] = []
    test_atol: float = 0.0
    test_rtol: float = 0.0
    bench_shapes: list[tuple] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_optype(cls)

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
    def op_type(self) -> str:
        """Default: same as name. Override if different."""
        return self.name

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

    @staticmethod
    @abstractmethod
    def test_fn(*args):
        """Function to compile with MegaKittens and use as reference for correctness tests."""
        ...

    @abstractmethod
    def test_args(self, shape: tuple) -> tuple:
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
