from enum import Enum


class OpType(str, Enum):
    """
    Operation type enum. Structural types (input, output) are defined here;
    all others are auto-registered from IType subclasses via __init_subclass__.
    """
    input = "input"
    output = "output"

    @classmethod
    def _missing_(cls, value):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._name_ = value
        return obj

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


_CALL_FUNCTION_MAP: dict = {}
_CALL_METHOD_MAP: dict[str, OpType] = {}
_CALL_MODULE_MAP: dict[type, OpType] = {}


def register_optype(cls):
    """Called by IType.__init_subclass__ to register op mappings at class definition time."""
    instance = cls()
    op = OpType(instance.op_type)
    for fn in cls.torch_functions:
        _CALL_FUNCTION_MAP[fn] = op
    for method in cls.torch_methods:
        _CALL_METHOD_MAP[method] = op
    for mod in cls.torch_modules:
        _CALL_MODULE_MAP[mod] = op
