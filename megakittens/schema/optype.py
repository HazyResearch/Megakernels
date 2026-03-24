import operator
from enum import Enum

import torch


class OpType(str, Enum):
    input = "input"
    add = "add"
    gemm = "gemm"
    relu = "relu"
    output = "output"

    @classmethod
    def from_call_function(cls, target) -> "OpType":
        if target not in _CALL_FUNCTION_MAP:
            raise ValueError(f"[MegaKittens] Unsupported call_function target: {target!r}")
        return _CALL_FUNCTION_MAP[target]

    @classmethod
    def from_call_method(cls, target: str) -> "OpType":
        if target not in _CALL_METHOD_MAP:
            raise ValueError(f"[MegaKittens] Unsupported call_method target: {target!r}")
        return _CALL_METHOD_MAP[target]

    @classmethod
    def from_call_module(cls, module_type: type) -> "OpType":
        if module_type not in _CALL_MODULE_MAP:
            raise ValueError(f"[MegaKittens] Unsupported module type: {module_type.__name__}")
        return _CALL_MODULE_MAP[module_type]


_CALL_FUNCTION_MAP = {
    torch.add: OpType.add,
    torch.matmul: OpType.gemm,
    torch.mm: OpType.gemm,
    torch.relu: OpType.relu,
    operator.add: OpType.add,
    operator.matmul: OpType.gemm,
    torch.ops.aten.add: OpType.add,
    torch.ops.aten.add.default: OpType.add,
    torch.ops.aten.add.Tensor: OpType.add,
    torch.ops.aten.mm: OpType.gemm,
    torch.ops.aten.mm.default: OpType.gemm,
    torch.ops.aten.matmul: OpType.gemm,
    torch.ops.aten.matmul.default: OpType.gemm,
    torch.ops.aten.relu: OpType.relu,
    torch.ops.aten.relu.default: OpType.relu,
}

_CALL_METHOD_MAP: dict[str, OpType] = {
    "add": OpType.add,
    "gemm": OpType.gemm,
    "relu": OpType.relu,
}

_CALL_MODULE_MAP: dict[type, OpType] = {
    torch.nn.ReLU: OpType.relu,
    torch.nn.ReLU6: OpType.relu,
}
