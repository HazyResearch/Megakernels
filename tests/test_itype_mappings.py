"""Note that this test is incomprehensive. Consider it a sanity check."""

import sys

import torch

import megakittens
from .common import check


def collect_mappings(names=None):
    mappings = []
    for cls in megakittens.schema.itype.IType.__subclasses__():
        if not cls.test_cases:
            continue
        inst = cls()
        tensors = tuple(
            torch.randint(0, 10, spec.granularity, dtype=spec.dtype.torch_dtype, device="cuda")
            if spec.dtype.torch_dtype in (torch.int32, torch.int64)
            else torch.randn(*spec.granularity, dtype=spec.dtype.torch_dtype, device="cuda").abs() + 1e-3
            for spec in inst.inputs
        )

        for key in cls.torch_functions_map:
            key_str = str(key)
            if "megakittens" in key_str or "aten." in key_str:
                continue
            try:
                key(*tensors)
            except (TypeError, RuntimeError):
                continue
            desc = f"{cls.__name__}:func:{key_str}"
            if names and desc not in names:
                continue
            mappings.append((lambda *a, _fn=key: _fn(*a), tensors, inst.test_atol, inst.test_rtol, desc))

        for method_name in cls.torch_methods_map:
            if not hasattr(tensors[0], method_name):
                continue
            try:
                getattr(tensors[0], method_name)(*tensors[1:])
            except (TypeError, RuntimeError):
                continue
            desc = f"{cls.__name__}:method:{method_name}"
            if names and desc not in names:
                continue
            mappings.append((lambda *a, _m=method_name: getattr(a[0], _m)(*a[1:]), tensors, inst.test_atol, inst.test_rtol, desc))

        for module_cls in cls.torch_modules_map:
            try:
                module = module_cls()
                module(*tensors)
            except (TypeError, RuntimeError):
                continue
            desc = f"{cls.__name__}:module:{module_cls.__name__}"
            if names and desc not in names:
                continue
            mappings.append((lambda *a, _m=module: _m(*a), tensors, inst.test_atol, inst.test_rtol, desc))

    return mappings


try:
    import pytest
    @pytest.mark.parametrize(
        "fn, tensors, atol, rtol, desc",
        collect_mappings(),
        ids=[desc for _, _, _, _, desc in collect_mappings()],
    )
    def test_itype_mapping(fn, tensors, atol, rtol, desc):
        check(fn, tensors, atol=atol, rtol=rtol)
except ImportError:
    pass


if __name__ == "__main__":
    names = sys.argv[1:] or None
    for fn, tensors, atol, rtol, desc in collect_mappings(names):
        max_diff, mean_diff = check(fn, tensors, atol=atol, rtol=rtol)
        print(f"  PASS {desc} | max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
