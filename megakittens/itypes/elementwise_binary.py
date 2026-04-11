import operator
from typing import List, Tuple

import torch

from ..dispatcher import Dispatcher
from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorSpec
from ..jit.pykittens import st


@torch.library.custom_op("megakittens::elementwise_binary", mutates_args=())
def elementwise_binary_op(tensors: list[torch.Tensor], ops: str) -> torch.Tensor:
    op_list = ops.split(",")  # Torch custom ops do not support list of strings
    result = ElementwiseBinary.BINARY_OPS[op_list[0]][1](tensors[0], tensors[1])
    for i, op in enumerate(op_list[1:], 2):
        result = ElementwiseBinary.BINARY_OPS[op][1](result, tensors[i])
    return result

@elementwise_binary_op.register_fake
def _elementwise_binary_fake(tensors: list[torch.Tensor], ops: str) -> torch.Tensor:
    return torch.empty_like(tensors[0])


def _resolve_from_custom_op(args, kwargs):
    ops_str = args[1] if len(args) > 1 else kwargs.get("ops", "add")
    return ElementwiseBinary(ops=tuple(ops_str.split(",")))

def _resolve_add(args, kwargs):
    return ElementwiseBinary(ops=("add",))

def _resolve_sub(args, kwargs):
    return ElementwiseBinary(ops=("sub",))

def _resolve_mul(args, kwargs):
    return ElementwiseBinary(ops=("mul",))

def _resolve_div(args, kwargs):
    return ElementwiseBinary(ops=("div",))

def _resolve_max(args, kwargs):
    return ElementwiseBinary(ops=("max",))

def _resolve_min(args, kwargs):
    return ElementwiseBinary(ops=("min",))


class ElementwiseBinary(IType):
    TILE_SIZE = 128
    TMA = st(dtype=DType.bf16, rows=128, cols=128)

    BINARY_OPS = {
        "add": ("BinaryOp::ADD", torch.add),
        "sub": ("BinaryOp::SUB", torch.sub),
        "mul": ("BinaryOp::MUL", torch.mul),
        "div": ("BinaryOp::DIV", torch.div),
        "max": ("BinaryOp::MAX", torch.maximum),
        "min": ("BinaryOp::MIN", torch.minimum),
    }

    torch_functions_map = {
        torch.ops.megakittens.elementwise_binary: _resolve_from_custom_op,
        torch.ops.megakittens.elementwise_binary.default: _resolve_from_custom_op,
        torch.add: _resolve_add, operator.add: _resolve_add,
        torch.ops.aten.add: _resolve_add, torch.ops.aten.add.default: _resolve_add,
        torch.ops.aten.add.Tensor: _resolve_add,
        torch.sub: _resolve_sub, operator.sub: _resolve_sub,
        torch.ops.aten.sub: _resolve_sub, torch.ops.aten.sub.default: _resolve_sub,
        torch.ops.aten.sub.Tensor: _resolve_sub,
        torch.mul: _resolve_mul, operator.mul: _resolve_mul,
        torch.ops.aten.mul: _resolve_mul, torch.ops.aten.mul.default: _resolve_mul,
        torch.ops.aten.mul.Tensor: _resolve_mul,
        torch.div: _resolve_div, operator.truediv: _resolve_div,
        torch.ops.aten.div: _resolve_div, torch.ops.aten.div.default: _resolve_div,
        torch.ops.aten.div.Tensor: _resolve_div,
        torch.maximum: _resolve_max,
        torch.ops.aten.maximum: _resolve_max, torch.ops.aten.maximum.default: _resolve_max,
        torch.minimum: _resolve_min,
        torch.ops.aten.minimum: _resolve_min, torch.ops.aten.minimum.default: _resolve_min,
    }
    torch_methods_map = {
        "add": _resolve_add, "sub": _resolve_sub,
        "mul": _resolve_mul, "div": _resolve_div,
    }

    test_cases = [
        (((op,),), shape)
        for op in BINARY_OPS.keys()
        for shape in [(128, 128), (512, 1024), (1280, 2048)]
    ] + [
        ((ops,), shape)
        for ops in [
            ("add", "add"),                                         # 2 ops, 3 inputs
            ("mul", "add"), ("sub", "div"), ("max", "min"),
            ("add", "mul", "sub"),                                  # 3 ops, 4 inputs
            ("add", "sub", "mul", "div"),                           # 4 ops, 5 inputs
            ("add", "sub", "mul", "max", "min"),                    # 5 ops, 6 inputs
            ("add", "sub", "mul", "div", "max", "min"),             # 6 ops, 7 inputs
        ]
        for shape in [(128, 128), (512, 1024), (1280, 2048)]
    ]
    test_atol = 0.2
    test_rtol = 1e-2
    bench_cases = [((("add",),), (4096, 4096)), ((("add",),), (131072, 4096)), ((("add",),), (4096, 131072)), ((("add",),), (16384, 16384)), ((("add",),), (131072, 131072))]

    def __init__(self, ops: tuple[str, ...] = ("add",)):
        self.ops = ops

    @property
    def num_inputs(self) -> int:
        return len(self.ops) + 1

    @property
    def tiles_per_inst(self) -> int:
        return Dispatcher.NUM_PAGES // self.num_inputs

    @staticmethod
    def test_fn(tensors, ops_str):
        return torch.ops.megakittens.elementwise_binary(tensors, ops_str)

    def test_args(self, case: tuple) -> tuple:
        M, N = case
        tensors = []
        for i in range(self.num_inputs):
            t = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
            if i >= 1 and self.ops[i - 1] == "div":
                t = t.abs() + 1e-3
            tensors.append(t)
        return (tensors, ",".join(self.ops))

    def bench_bytes(self, case: tuple) -> float:
        M, N = case
        return M * N * 2 * (self.num_inputs + 1)

    @property
    def cpp_template(self) -> str:
        ops_str = ", ".join(self.BINARY_OPS[op][0] for op in self.ops)
        return f"ElementwiseBinary<MKConfig, MKGlobals, BinaryOpList<{ops_str}>, {{tensors}}>"

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.TILE_SIZE, self.TILE_SIZE), tma_types=[self.TMA])
            for _ in range(self.num_inputs)
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.TILE_SIZE, self.TILE_SIZE), tma_types=[self.TMA]),
        ]

    def block_indices(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> List[Tuple[int, ...]]:
        rows = dst_metas[0].shape[0] // self.TILE_SIZE
        cols = dst_metas[0].shape[1] // self.TILE_SIZE
        indices = []
        for row in range(rows):
            for col in range(0, cols, self.tiles_per_inst):
                n = min(self.tiles_per_inst, cols - col)
                indices.append((row, col, n))
        return indices

    def num_instructions(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> int:
        rows = dst_metas[0].shape[0] // self.TILE_SIZE
        cols = dst_metas[0].shape[1] // self.TILE_SIZE
        return rows * ((cols + self.tiles_per_inst - 1) // self.tiles_per_inst)

    def access_regions(self, block_index, src_metas, dst_metas):
        row, col, n = block_index
        region = ((row * self.TILE_SIZE, (row + 1) * self.TILE_SIZE), (col * self.TILE_SIZE, (col + n) * self.TILE_SIZE))
        return [region] * self.num_inputs, [region]

    def validate(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> None:
        super().validate(src_metas, dst_metas)
        for op in self.ops:
            if op not in self.BINARY_OPS:
                raise RuntimeError(f"[MegaKittens] ElementwiseBinary: unknown op {op!r}")
        shapes = [m.shape for m in src_metas]
        if len(set(shapes)) > 1:
            raise RuntimeError(
                f"[MegaKittens] ElementwiseBinary requires same-shape inputs, got {shapes}. "
                f"Broadcasting is not supported."
            )
