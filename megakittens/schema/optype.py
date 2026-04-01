import operator
from enum import Enum

import torch


class OpType(str, Enum):
    noop = "noop"
    input = "input"
    add = "add"
    gemm = "gemm"
    relu = "relu"
    rmsnorm = "rmsnorm"
    rms_lm_head = "rms_lm_head"
    attention = "attention"
    causal_attention = "causal_attention"
    rms_upgate_silu = "rms_upgate_silu"
    proj_residual = "proj_residual"
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
    torch.ops.megakittens.rmsnorm: OpType.rmsnorm,
    torch.ops.megakittens.rmsnorm.default: OpType.rmsnorm,
    torch.ops.megakittens.rms_lm_head: OpType.rms_lm_head,
    torch.ops.megakittens.rms_lm_head.default: OpType.rms_lm_head,
    torch.ops.megakittens.rms_upgate_silu: OpType.rms_upgate_silu,
    torch.ops.megakittens.rms_upgate_silu.default: OpType.rms_upgate_silu,
    torch.ops.megakittens.proj_residual: OpType.proj_residual,
    torch.ops.megakittens.proj_residual.default: OpType.proj_residual,
    torch.ops.megakittens.attention: OpType.attention,
    torch.ops.megakittens.attention.default: OpType.attention,
    torch.ops.megakittens.causal_attention: OpType.causal_attention,
    torch.ops.megakittens.causal_attention.default: OpType.causal_attention,
}

_CALL_METHOD_MAP: dict[str, OpType] = {
    "add": OpType.add,
    "gemm": OpType.gemm,
    "relu": OpType.relu,
    "rmsnorm": OpType.rmsnorm,
    "attention": OpType.attention,
    "causal_attention": OpType.causal_attention,
}

_CALL_MODULE_MAP: dict[type, OpType] = {
    torch.nn.ReLU: OpType.relu,
    torch.nn.ReLU6: OpType.relu,
    torch.nn.RMSNorm: OpType.rmsnorm,
}
