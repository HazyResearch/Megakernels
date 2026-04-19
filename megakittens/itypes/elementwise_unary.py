from typing import List, Tuple

import torch

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


def _resolve_from_custom_op(args, kwargs):
    ops_str = args[1] if len(args) > 1 else kwargs.get("ops", "relu")
    return ElementwiseUnary(ops=tuple(ops_str.split(",")))

def _resolve_identity(args, kwargs):
    return ElementwiseUnary(ops=("identity",))

def _resolve_relu(args, kwargs):
    return ElementwiseUnary(ops=("relu",))

def _resolve_abs(args, kwargs):
    return ElementwiseUnary(ops=("abs",))

def _resolve_exp(args, kwargs):
    return ElementwiseUnary(ops=("exp",))

def _resolve_exp2(args, kwargs):
    return ElementwiseUnary(ops=("exp2",))

def _resolve_log(args, kwargs):
    return ElementwiseUnary(ops=("log",))

def _resolve_log2(args, kwargs):
    return ElementwiseUnary(ops=("log2",))

def _resolve_neg(args, kwargs):
    return ElementwiseUnary(ops=("neg",))

def _resolve_sqrt(args, kwargs):
    return ElementwiseUnary(ops=("sqrt",))

def _resolve_rsqrt(args, kwargs):
    return ElementwiseUnary(ops=("rsqrt",))


class ElementwiseUnary(IType):
    TILE_SIZE = 128
    MAX_TILES_PER_INST = 2
    TMA = st(dtype=DType.bf16, rows=128, cols=128)

    UNARY_OPS = {
        "identity": ("UnaryOp::IDENTITY", torch.clone),
        "relu":     ("UnaryOp::RELU",     torch.relu),
        "abs":      ("UnaryOp::ABS",      torch.abs),
        "exp":      ("UnaryOp::EXP",      torch.exp),
        "exp2":     ("UnaryOp::EXP2",     torch.exp2),
        "log":      ("UnaryOp::LOG",      torch.log),
        "log2":     ("UnaryOp::LOG2",     torch.log2),
        "neg":      ("UnaryOp::NEG",      torch.neg),
        "sqrt":     ("UnaryOp::SQRT",     torch.sqrt),
        "rsqrt":    ("UnaryOp::RSQRT",    torch.rsqrt),
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

    def __init__(self, ops: tuple[str, ...] = ("relu",)):
        self.ops = ops

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
        return f"ElementwiseUnary<MKConfig, MKGlobals, {{tensors}}, {ops_str}>"

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.TILE_SIZE, self.TILE_SIZE), tma_types=[self.TMA]),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.TILE_SIZE, self.TILE_SIZE), tma_types=[self.TMA]),
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
        indices = []
        for b in range(dst_range[0].size):
            for d in range(dst_range[1].size):
                for r in range(dst_range[2].size // self.TILE_SIZE):
                    for c in range(0, dst_range[3].size // self.TILE_SIZE, self.MAX_TILES_PER_INST):
                        n = min(self.MAX_TILES_PER_INST, dst_range[3].size // self.TILE_SIZE - c)
                        indices.append((
                            src_range[0].start + b, src_range[1].start + d, src_range[2].start // self.TILE_SIZE + r, src_range[3].start // self.TILE_SIZE + c,
                            dst_range[0].start + b, dst_range[1].start + d, dst_range[2].start // self.TILE_SIZE + r, dst_range[3].start // self.TILE_SIZE + c,
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
        return dst_range[0].size * dst_range[1].size * (dst_range[2].size // self.TILE_SIZE) * ((dst_range[3].size // self.TILE_SIZE + self.MAX_TILES_PER_INST - 1) // self.MAX_TILES_PER_INST)

    def access_regions(
        self,
        block_index: Tuple[int, ...],
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ) -> tuple[list[list[tuple[tuple[int, int], ...]]], list[list[tuple[tuple[int, int], ...]]]]:
        b_src, d_src, r_src, c_src, b_dst, d_dst, r_dst, c_dst, n = block_index
        src_region = ((b_src, b_src + 1), (d_src, d_src + 1),
                      (r_src * self.TILE_SIZE, (r_src + 1) * self.TILE_SIZE),
                      (c_src * self.TILE_SIZE, (c_src + n) * self.TILE_SIZE))
        dst_region = ((b_dst, b_dst + 1), (d_dst, d_dst + 1),
                      (r_dst * self.TILE_SIZE, (r_dst + 1) * self.TILE_SIZE),
                      (c_dst * self.TILE_SIZE, (c_dst + n) * self.TILE_SIZE))
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
