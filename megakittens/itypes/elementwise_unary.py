from typing import List, Tuple

import torch

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorSpec
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
    TILES_PER_INST = 7
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
    torch_modules_map = {torch.nn.ReLU: _resolve_relu, torch.nn.ReLU6: _resolve_relu}

    test_cases = [
        (((op,),), shape)
        for op in UNARY_OPS.keys()
        for shape in [(128, 128), (512, 1024), (1280, 2048)]
    ] + [
        ((ops,), shape)
        for ops in [("abs", "neg"), ("neg", "abs"), ("exp", "log"), ("relu", "sqrt"), ("abs", "log", "neg")]
        for shape in [(128, 128), (512, 1024), (1280, 2048)]
    ]
    test_atol = 1e-2
    test_rtol = 1e-2
    bench_cases = [((("relu",),), (4096, 4096)), ((("relu",),), (131072, 4096)), ((("relu",),), (4096, 131072)), ((("relu",),), (16384, 16384)), ((("relu",),), (131072, 131072)),]

    def __init__(self, ops: tuple[str, ...] = ("relu",)):
        self.ops = ops

    def test_args(self, case: tuple) -> tuple:
        M, N = case
        x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
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

    def block_indices(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> List[Tuple[int, ...]]:
        rows = dst_metas[0].shape[0] // self.TILE_SIZE
        cols = dst_metas[0].shape[1] // self.TILE_SIZE
        indices = []
        for row in range(rows):
            for col in range(0, cols, self.TILES_PER_INST):
                n = min(self.TILES_PER_INST, cols - col)
                indices.append((row, col, n))
        return indices

    def num_instructions(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> int:
        rows = dst_metas[0].shape[0] // self.TILE_SIZE
        cols = dst_metas[0].shape[1] // self.TILE_SIZE
        return rows * ((cols + self.TILES_PER_INST - 1) // self.TILES_PER_INST)

    def access_regions(self, block_index, src_metas, dst_metas):
        row, col, n = block_index
        region = ((row * self.TILE_SIZE, (row + 1) * self.TILE_SIZE), (col * self.TILE_SIZE, (col + n) * self.TILE_SIZE))
        return [region], [region]

    def validate(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> None:
        super().validate(src_metas, dst_metas)
        for op in self.ops:
            if op not in self.UNARY_OPS:
                raise RuntimeError(f"[MegaKittens] ElementwiseUnary: unknown op {op!r}")
        if src_metas[0].shape != dst_metas[0].shape:
            raise RuntimeError(
                f"[MegaKittens] ElementwiseUnary output shape {dst_metas[0].shape} doesn't match input shape {src_metas[0].shape}"
            )
