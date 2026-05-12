import struct
from typing import List, Tuple

import torch

from ..dispatcher import Dispatcher
from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorRange, TensorSpec
from ..jit.pykittens import st


@torch.library.custom_op("megakittens::elementwise_unary", mutates_args=())
def elementwise_unary_op(x: torch.Tensor, ops: str) -> torch.Tensor:
    for op in ops.split(","):  # Torch custom ops do not support list of strings
        x = ElementwiseUnary.UNARY_OPS[op][1](x)
    return x


@elementwise_unary_op.register_fake
def _elementwise_unary_fake(x: torch.Tensor, ops: str) -> torch.Tensor:
    return torch.empty_like(x)


def _detect_dtype(args):
    if args and hasattr(args[0], 'meta') and 'val' in args[0].meta:
        val = args[0].meta['val']
        if isinstance(val, torch.Tensor):
            return DType.from_torch(val.dtype)
    raise RuntimeError(f"[MegaKittens] ElementwiseUnary: cannot detect dtype from args")


def _resolve_from_custom_op(args, kwargs):
    ops_str = args[1] if len(args) > 1 else kwargs.get("ops", "relu")
    return ElementwiseUnary(ops=tuple(ops_str.split(",")))

def _resolve_identity(args, kwargs):
    return ElementwiseUnary(ops=("identity",), dtype=_detect_dtype(args))

def _resolve_relu(args, kwargs):
    return ElementwiseUnary(ops=("relu",), dtype=_detect_dtype(args))

def _resolve_abs(args, kwargs):
    return ElementwiseUnary(ops=("abs",), dtype=_detect_dtype(args))

def _resolve_exp(args, kwargs):
    return ElementwiseUnary(ops=("exp",), dtype=_detect_dtype(args))

def _resolve_exp2(args, kwargs):
    return ElementwiseUnary(ops=("exp2",), dtype=_detect_dtype(args))

def _resolve_log(args, kwargs):
    return ElementwiseUnary(ops=("log",), dtype=_detect_dtype(args))

def _resolve_log2(args, kwargs):
    return ElementwiseUnary(ops=("log2",), dtype=_detect_dtype(args))

def _resolve_neg(args, kwargs):
    return ElementwiseUnary(ops=("neg",), dtype=_detect_dtype(args))

def _resolve_sqrt(args, kwargs):
    return ElementwiseUnary(ops=("sqrt",), dtype=_detect_dtype(args))

def _resolve_rsqrt(args, kwargs):
    return ElementwiseUnary(ops=("rsqrt",), dtype=_detect_dtype(args))


class ElementwiseUnary(IType):
    TILE_ROWS = 128
    MAX_TILES_PER_INST = 2

    UNARY_OPS = {
        "identity":   ("UnaryOp::IDENTITY",   torch.clone),
        "relu":       ("UnaryOp::RELU",       torch.relu),
        "abs":        ("UnaryOp::ABS",        torch.abs),
        "exp":        ("UnaryOp::EXP",        torch.exp),
        "exp2":       ("UnaryOp::EXP2",       torch.exp2),
        "log":        ("UnaryOp::LOG",        torch.log),
        "log2":       ("UnaryOp::LOG2",       torch.log2),
        "neg":        ("UnaryOp::NEG",        torch.neg),
        "sqrt":       ("UnaryOp::SQRT",       torch.sqrt),
        "rsqrt":      ("UnaryOp::RSQRT",      torch.rsqrt),
        "add_scalar": ("UnaryOp::ADD_SCALAR", None),
        "mul_scalar": ("UnaryOp::MUL_SCALAR", None),
        "sub_scalar": ("UnaryOp::SUB_SCALAR", None),
        "div_scalar": ("UnaryOp::DIV_SCALAR", None),
        "max_scalar": ("UnaryOp::MAX_SCALAR", None),
        "min_scalar": ("UnaryOp::MIN_SCALAR", None),
    }

    torch_functions_map = {
        torch.ops.megakittens.elementwise_unary: _resolve_from_custom_op,
        torch.ops.megakittens.elementwise_unary.default: _resolve_from_custom_op,
        torch.clone: _resolve_identity,
        torch.ops.aten.clone: _resolve_identity, torch.ops.aten.clone.default: _resolve_identity,
        torch.relu: _resolve_relu,
        torch.ops.aten.relu: _resolve_relu, torch.ops.aten.relu.default: _resolve_relu,
        torch.abs: _resolve_abs,
        torch.ops.aten.abs: _resolve_abs, torch.ops.aten.abs.default: _resolve_abs,
        torch.exp: _resolve_exp,
        torch.ops.aten.exp: _resolve_exp, torch.ops.aten.exp.default: _resolve_exp,
        torch.exp2: _resolve_exp2,
        torch.ops.aten.exp2: _resolve_exp2, torch.ops.aten.exp2.default: _resolve_exp2,
        torch.log: _resolve_log,
        torch.ops.aten.log: _resolve_log, torch.ops.aten.log.default: _resolve_log,
        torch.log2: _resolve_log2,
        torch.ops.aten.log2: _resolve_log2, torch.ops.aten.log2.default: _resolve_log2,
        torch.neg: _resolve_neg,
        torch.ops.aten.neg: _resolve_neg, torch.ops.aten.neg.default: _resolve_neg,
        torch.sqrt: _resolve_sqrt,
        torch.ops.aten.sqrt: _resolve_sqrt, torch.ops.aten.sqrt.default: _resolve_sqrt,
        torch.rsqrt: _resolve_rsqrt,
        torch.ops.aten.rsqrt: _resolve_rsqrt, torch.ops.aten.rsqrt.default: _resolve_rsqrt,
    }
    torch_methods_map = {
        "relu": _resolve_relu, "abs": _resolve_abs,
        "exp": _resolve_exp, "exp2": _resolve_exp2,
        "log": _resolve_log, "log2": _resolve_log2,
        "neg": _resolve_neg, "sqrt": _resolve_sqrt, "rsqrt": _resolve_rsqrt,
    }
    torch_modules_map = {torch.nn.ReLU: _resolve_relu}

    test_cases = [
        (((op,),), shape)
        for op in UNARY_OPS.keys()
        for shape in [(128, 128), (512, 1024), (1280, 2048), (2, 128, 256), (3, 512, 1024), (2, 3, 128, 256)]
    ] + [
        ((ops,), shape)
        for ops in [("abs", "neg"), ("neg", "abs"), ("exp", "log"), ("relu", "sqrt"), ("abs", "log", "neg")]
        for shape in [(128, 128), (512, 1024), (1280, 2048), (2, 128, 256), (3, 512, 1024), (2, 3, 128, 256)]
    ]
    test_atol = 1e-2
    test_rtol = 1e-2
    bench_cases = [((("relu",),), (4096, 4096)), ((("relu",),), (131072, 4096)), ((("relu",),), (4096, 131072)), ((("relu",),), (16384, 16384)), ((("relu",),), (131072, 131072)),]

    def __init__(self, ops: tuple[str, ...] = ("relu",), dtype: DType = DType.bf16, scalar_val: float | None = None):
        self.ops = ops
        self.dtype = dtype
        self.scalar_val = scalar_val

    @property
    def tile_cols(self) -> int:
        return Dispatcher.PAGE_SIZE // (self.TILE_ROWS * self.dtype.size)

    def test_args(self, case: tuple) -> tuple:
        x = torch.randn(*case, dtype=torch.bfloat16, device="cuda")
        if any(op in ("log", "log2", "sqrt", "rsqrt") for op in self.ops):
            x = x.abs() + 1e-3  # ensure positive for log/sqrt/rsqrt
        return (x, ",".join(self.ops))

    def bench_bytes(self, case: tuple) -> float:
        M, N = case
        return M * N * 2 * 2  # 2 bytes/bf16, 2 tensors (1 read + 1 write)

    @property
    def cpp_template(self) -> str:
        ops_str = ", ".join(self.UNARY_OPS[op][0] for op in self.ops)
        scalar_bits = 0
        if self.scalar_val is not None:
            if self.dtype == DType.fp32:
                scalar_bits = struct.unpack('I', struct.pack('f', self.scalar_val))[0]
            elif self.dtype == DType.bf16:
                scalar_bits = torch.tensor(self.scalar_val, dtype=torch.bfloat16, device='cpu').view(torch.int16).item() & 0xFFFF
            elif self.dtype == DType.half:
                scalar_bits = struct.unpack('H', struct.pack('e', self.scalar_val))[0]
            else:
                raise RuntimeError(f"[MegaKittens] Unsupported dtype for scalar op: {self.dtype}")
        return f"ElementwiseUnary<MKConfig, MKGlobals, {self.dtype.cpp_dtype}, {{tensors}}, {scalar_bits}u, {ops_str}>"

    @property
    def inputs(self) -> list[TensorSpec]:
        tma = st(dtype=self.dtype, rows=self.TILE_ROWS, cols=self.tile_cols)
        return [
            TensorSpec(dtype=self.dtype, granularity=(self.TILE_ROWS, self.tile_cols), tma_types=[tma]),
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
        src_range = src_ranges[0]
        dst_range = dst_ranges[0]
        tr, tc = self.TILE_ROWS, self.tile_cols
        indices = []
        for b in range(dst_range[0].size):
            for d in range(dst_range[1].size):
                for r in range(dst_range[2].size // tr):
                    for c in range(0, dst_range[3].size // tc, self.MAX_TILES_PER_INST):
                        n = min(self.MAX_TILES_PER_INST, dst_range[3].size // tc - c)
                        indices.append((
                            src_range[0].start + b, src_range[1].start + d, src_range[2].start // tr + r, src_range[3].start // tc + c,
                            dst_range[0].start + b, dst_range[1].start + d, dst_range[2].start // tr + r, dst_range[3].start // tc + c,
                            n,
                        ))
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
        return dst_range[0].size * dst_range[1].size * (dst_range[2].size // tr) * ((dst_range[3].size // tc + self.MAX_TILES_PER_INST - 1) // self.MAX_TILES_PER_INST)

    def access_regions(
        self,
        block_index: Tuple[int, ...],
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ) -> tuple[list[list[tuple[tuple[int, int], ...]]], list[list[tuple[tuple[int, int], ...]]]]:
        b_src, d_src, r_src, c_src, b_dst, d_dst, r_dst, c_dst, n = block_index
        tr, tc = self.TILE_ROWS, self.tile_cols
        src_region = ((b_src, b_src + 1), (d_src, d_src + 1),
                      (r_src * tr, (r_src + 1) * tr),
                      (c_src * tc, (c_src + n) * tc))
        dst_region = ((b_dst, b_dst + 1), (d_dst, d_dst + 1),
                      (r_dst * tr, (r_dst + 1) * tr),
                      (c_dst * tc, (c_dst + n) * tc))
        return [[src_region]], [[dst_region]]

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        for op in self.ops:
            if op not in self.UNARY_OPS:
                raise RuntimeError(f"[MegaKittens] ElementwiseUnary: unknown op {op!r}")
        if src_ranges[0].effective_shape != dst_ranges[0].effective_shape:
            raise RuntimeError(
                f"[MegaKittens] ElementwiseUnary effective shape mismatch: "
                f"src={src_ranges[0].effective_shape} dst={dst_ranges[0].effective_shape}"
            )
