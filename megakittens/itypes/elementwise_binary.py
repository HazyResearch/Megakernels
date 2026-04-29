import operator
from typing import List, Tuple

import torch

from ..dispatcher import Dispatcher
from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorRange, TensorSpec
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


def _detect_dtype(args):
    dtypes = set()
    for arg in args:
        nodes = arg if isinstance(arg, (list, tuple)) else [arg]
        for node in nodes:
            if hasattr(node, 'meta') and 'val' in node.meta:
                val = node.meta['val']
                if isinstance(val, torch.Tensor):
                    dtypes.add(val.dtype)
    if len(dtypes) > 1:
        raise RuntimeError(f"[MegaKittens] ElementwiseBinary: mixed dtypes among operands: {dtypes}")
    if len(dtypes) == 1:
        return DType.from_torch(next(iter(dtypes)))
    raise RuntimeError(f"[MegaKittens] ElementwiseBinary: cannot detect dtype from args")

def _resolve_from_custom_op(args, kwargs):
    ops_str = args[1] if len(args) > 1 else kwargs.get("ops", "add")
    return ElementwiseBinary(ops=tuple(ops_str.split(",")))

def _resolve_add(args, kwargs):
    return ElementwiseBinary(ops=("add",), dtype=_detect_dtype(args))

def _resolve_sub(args, kwargs):
    return ElementwiseBinary(ops=("sub",), dtype=_detect_dtype(args))

def _resolve_mul(args, kwargs):
    return ElementwiseBinary(ops=("mul",), dtype=_detect_dtype(args))

def _resolve_div(args, kwargs):
    return ElementwiseBinary(ops=("div",), dtype=_detect_dtype(args))

def _resolve_max(args, kwargs):
    return ElementwiseBinary(ops=("max",), dtype=_detect_dtype(args))

def _resolve_min(args, kwargs):
    return ElementwiseBinary(ops=("min",), dtype=_detect_dtype(args))



class ElementwiseBinary(IType):
    MAX_OPS = 1
    TILE_ROWS = 128
    MAX_TILES_PER_INST = 2

    BINARY_OPS = {
        "add":   ("BinaryOp::ADD",   torch.add),
        "sub":   ("BinaryOp::SUB",   torch.sub),
        "mul":   ("BinaryOp::MUL",   torch.mul),
        "div":   ("BinaryOp::DIV",   torch.div),
        "max":   ("BinaryOp::MAX",   torch.maximum),
        "min":   ("BinaryOp::MIN",   torch.minimum),
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
        for shape in [(128, 128), (512, 1024), (1280, 2048), (2, 128, 256), (3, 512, 1024), (2, 3, 128, 256)]
    ] + [
        # Disabled for now, as the current MAX_OPS is 1
        # ((ops,), shape)
        # for ops in [
        #     ("add", "add"),                                         # 2 ops, 3 inputs
        #     ("mul", "add"), ("sub", "div"), ("max", "min"),
        #     ("add", "mul", "sub"),                                  # 3 ops, 4 inputs
        #     ("add", "sub", "mul", "div"),                           # 4 ops, 5 inputs
        #     ("add", "sub", "mul", "max", "min"),                    # 5 ops, 6 inputs
        #     ("add", "sub", "mul", "div", "max", "min"),             # 6 ops, 7 inputs
        # ]
        # for shape in [(128, 128), (512, 1024), (1280, 2048), (2, 128, 256), (3, 512, 1024), (2, 3, 128, 256)]
    ]
    test_atol = 0.2
    test_rtol = 1e-2
    bench_cases = [((("add",),), (4096, 4096)), ((("add",),), (131072, 4096)), ((("add",),), (4096, 131072)), ((("add",),), (16384, 16384)), ((("add",),), (131072, 131072))]

    def __init__(self, ops: tuple[str, ...] = ("add",), dtype: DType = DType.bf16):
        self.ops = ops
        self.dtype = dtype

    @property
    def tile_cols(self) -> int:
        return Dispatcher.PAGE_SIZE // (self.TILE_ROWS * self.dtype.size)

    @property
    def num_inputs(self) -> int:
        return len(self.ops) + 1

    @property
    def tiles_per_inst(self) -> int:
        return min(Dispatcher.NUM_PAGES // self.num_inputs, self.MAX_TILES_PER_INST)

    def test_args(self, case: tuple) -> tuple:
        tensors = []
        for i in range(self.num_inputs):
            t = torch.randn(*case, dtype=torch.bfloat16, device="cuda")
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
        return f"ElementwiseBinary<MKConfig, MKGlobals, {self.dtype.cpp_dtype}, BinaryOps<{ops_str}>, {{tensors}}>"

    @property
    def inputs(self) -> list[TensorSpec]:
        tma = st(dtype=self.dtype, rows=self.TILE_ROWS, cols=self.tile_cols)
        return [
            TensorSpec(dtype=self.dtype, granularity=(self.TILE_ROWS, self.tile_cols), tma_types=[tma])
            for _ in range(self.num_inputs)
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        tma = st(dtype=self.dtype, rows=self.TILE_ROWS, cols=self.tile_cols)
        return [
            TensorSpec(dtype=self.dtype, granularity=(self.TILE_ROWS, self.tile_cols), tma_types=[tma]),
        ]

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        dst_range = dst_ranges[0]
        tr, tc = self.TILE_ROWS, self.tile_cols
        indices = []
        for b in range(dst_range[0].size):
            for d in range(dst_range[1].size):
                for r in range(dst_range[2].size // tr):
                    for c in range(0, dst_range[3].size // tc, self.tiles_per_inst):
                        n = min(self.tiles_per_inst, dst_range[3].size // tc - c)
                        index: list[int] = []
                        for src_range in src_ranges:
                            index.extend([src_range[0].start + b, src_range[1].start + d, src_range[2].start // tr + r, src_range[3].start // tc + c])
                        index.extend([dst_range[0].start + b, dst_range[1].start + d, dst_range[2].start // tr + r, dst_range[3].start // tc + c])
                        index.append(n)
                        indices.append(tuple(index))
        return indices

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> int:
        dst_range = dst_ranges[0]
        tr, tc = self.TILE_ROWS, self.tile_cols
        return dst_range[0].size * dst_range[1].size * (dst_range[2].size // tr) * ((dst_range[3].size // tc + self.tiles_per_inst - 1) // self.tiles_per_inst)

    def access_regions(
        self,
        block_index: Tuple[int, ...],
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ) -> tuple[list[list[tuple[tuple[int, int], ...]]], list[list[tuple[tuple[int, int], ...]]]]:
        n = block_index[-1]
        tr, tc = self.TILE_ROWS, self.tile_cols
        src_regions = []
        for i in range(self.num_inputs):
            b_src, d_src, r_src, c_src = block_index[i * 4: i * 4 + 4]
            src_regions.append([(
                (b_src, b_src + 1), (d_src, d_src + 1),
                (r_src * tr, (r_src + 1) * tr),
                (c_src * tc, (c_src + n) * tc),
            )])
        b_dst, d_dst, r_dst, c_dst = block_index[self.num_inputs * 4: self.num_inputs * 4 + 4]
        dst_region = (
            (b_dst, b_dst + 1), (d_dst, d_dst + 1),
            (r_dst * tr, (r_dst + 1) * tr),
            (c_dst * tc, (c_dst + n) * tc),
        )
        return src_regions, [[dst_region]]

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        for op in self.ops:
            if op not in self.BINARY_OPS:
                raise RuntimeError(f"[MegaKittens] ElementwiseBinary: unknown op {op!r}")
        if len(self.ops) > self.MAX_OPS:
            raise RuntimeError(
                f"[MegaKittens] ElementwiseBinary supports at most {self.MAX_OPS} op(s) with per-tensor indices, got {len(self.ops)}"
            )
        for i, src_range in enumerate(src_ranges):
            if src_range.effective_shape != dst_ranges[0].effective_shape:
                raise RuntimeError(
                    f"[MegaKittens] ElementwiseBinary effective shape mismatch at src {i}: "
                    f"src={src_range.effective_shape} dst={dst_ranges[0].effective_shape}"
                )
