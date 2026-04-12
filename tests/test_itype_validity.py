import sys
from collections.abc import Iterable

import re

import torch

import megakittens


def collect_test_cases(names=None):
    test_cases = []
    for cls in megakittens.schema.itype.IType.__subclasses__():
        try:
            inst = cls()
        except TypeError:
            raise RuntimeError(f"{cls.__name__} must support a default constructor")
        if names and inst.name not in names:
            continue
        test_cases.append(inst)
    return test_cases


def check_validity(inst):
    cls = type(inst)

    # Class-level attributes

    assert isinstance(cls.torch_functions_map, dict), f"{cls.__name__}.torch_functions_map"
    for key, val in cls.torch_functions_map.items():
        assert callable(key), f"{cls.__name__}.torch_functions_map key {key!r} must be callable"
        assert val is None or callable(val), f"{cls.__name__}.torch_functions_map[{key!r}]"
    assert isinstance(cls.torch_methods_map, dict), f"{cls.__name__}.torch_methods_map"
    for key, val in cls.torch_methods_map.items():
        assert isinstance(key, str), f"{cls.__name__}.torch_methods_map key {key!r} must be str"
        assert val is None or callable(val), f"{cls.__name__}.torch_methods_map[{key!r}]"
    assert isinstance(cls.torch_modules_map, dict), f"{cls.__name__}.torch_modules_map"
    for key, val in cls.torch_modules_map.items():
        assert isinstance(key, type), f"{cls.__name__}.torch_modules_map key {key!r} must be a type"
        assert val is None or callable(val), f"{cls.__name__}.torch_modules_map[{key!r}]"
    assert isinstance(cls.test_cases, list), f"{cls.__name__}.test_cases must be list"
    for i, entry in enumerate(cls.test_cases):
        assert isinstance(entry, tuple) and len(entry) == 2, f"{cls.__name__}.test_cases[{i}]"
        cls_args, input_args = entry
        assert isinstance(cls_args, tuple), f"{cls.__name__}.test_cases[{i}][0] must be tuple"
        assert isinstance(input_args, tuple), f"{cls.__name__}.test_cases[{i}][1] must be tuple"
    assert isinstance(cls.test_atol, (int, float)) and cls.test_atol >= 0, f"{cls.__name__}.test_atol"
    assert isinstance(cls.test_rtol, (int, float)) and cls.test_rtol >= 0, f"{cls.__name__}.test_rtol"
    assert isinstance(cls.bench_cases, list), f"{cls.__name__}.bench_cases must be list"
    for i, entry in enumerate(cls.bench_cases):
        assert isinstance(entry, tuple) and len(entry) == 2, f"{cls.__name__}.bench_cases[{i}]"
        assert isinstance(entry[0], tuple), f"{cls.__name__}.bench_cases[{i}][0] must be tuple"
        assert isinstance(entry[1], tuple), f"{cls.__name__}.bench_cases[{i}][1] must be tuple"
    if cls.test_cases:
        assert hasattr(cls, "test_fn") and callable(cls.test_fn), f"{cls.__name__} needs test_fn"
        assert hasattr(cls, "test_args") and callable(cls.test_args), f"{cls.__name__} needs test_args"

    # Non-noop itypes must have at least 1 test case and 1 bench case

    if cls.__name__ != "Noop":
        assert len(cls.test_cases) >= 1, f"{cls.__name__} must have at least 1 test case"
        assert len(cls.bench_cases) >= 1, f"{cls.__name__} must have at least 1 bench case"

    # Property return types

    name = inst.name
    assert isinstance(name, str) and len(name) > 0, f"{cls.__name__}.name"
    cpp_template = inst.cpp_template
    assert isinstance(cpp_template, str) and len(cpp_template) > 0, f"{cls.__name__}.cpp_template"
    cpp_include = inst.cpp_include
    assert isinstance(cpp_include, str) and cpp_include.endswith(".cuh"), f"{cls.__name__}.cpp_include"
    inputs = inst.inputs
    assert isinstance(inputs, list), f"{cls.__name__}.inputs must be list"
    for i, spec in enumerate(inputs):
        assert isinstance(spec, megakittens.schema.tensor.TensorSpec), \
            f"{cls.__name__}.inputs[{i}] must be TensorSpec"
    outputs = inst.outputs
    assert isinstance(outputs, list), f"{cls.__name__}.outputs must be list"
    for i, spec in enumerate(outputs):
        assert isinstance(spec, megakittens.schema.tensor.TensorSpec), \
            f"{cls.__name__}.outputs[{i}] must be TensorSpec"

    # cpp_template contracts

    if inputs or outputs:
        assert "MKConfig" in cpp_template, f"{cls.__name__}.cpp_template missing MKConfig"
        assert "MKGlobals" in cpp_template, f"{cls.__name__}.cpp_template missing MKGlobals"
        assert "{tensors}" in cpp_template, f"{cls.__name__}.cpp_template missing {{tensors}}"

    # test_args return type

    for cls_args, input_args in cls.test_cases[:1]:
        result = cls(*cls_args).test_args(input_args)
        assert isinstance(result, tuple), f"{cls.__name__}.test_args must return tuple"

    # Method return types with dummy metas

    src_metas = tuple(megakittens.schema.tensor.TensorMeta(
        dtype=s.dtype, shape=tuple(g * 2 for g in s.granularity),
        device=megakittens.schema.device.Device(type="cuda", index=0),
    ) for s in inputs)
    dst_metas = tuple(megakittens.schema.tensor.TensorMeta(
        dtype=s.dtype, shape=tuple(g * 2 for g in s.granularity),
        device=megakittens.schema.device.Device(type="cuda", index=0),
    ) for s in outputs)

    try:
        block_idx_list = inst.block_indices(src_metas, dst_metas)
        assert isinstance(block_idx_list, list), f"{cls.__name__}.block_indices must return list"
        for i, idx in enumerate(block_idx_list):
            assert isinstance(idx, tuple), f"{cls.__name__}.block_indices()[{i}] must be tuple"
            for j, val in enumerate(idx):
                assert isinstance(val, int), f"{cls.__name__}.block_indices()[{i}][{j}] must be int"
        num_inst = inst.num_instructions(src_metas, dst_metas)
        assert isinstance(num_inst, int) and num_inst >= 0, f"{cls.__name__}.num_instructions"
        assert num_inst == len(block_idx_list), \
            f"{cls.__name__}.num_instructions()={num_inst} != len(block_indices())={len(block_idx_list)}"
        if block_idx_list:
            regions = inst.access_regions(block_idx_list[0], src_metas, dst_metas)
            assert isinstance(regions, tuple) and len(regions) == 2, \
                f"{cls.__name__}.access_regions must return 2-tuple"
            src_regions, dst_regions = regions
            assert isinstance(src_regions, list), f"{cls.__name__}.access_regions src must be list"
            assert isinstance(dst_regions, list), f"{cls.__name__}.access_regions dst must be list"
            assert len(src_regions) == len(inputs), \
                f"{cls.__name__}.access_regions src count mismatch"
            assert len(dst_regions) == len(outputs), \
                f"{cls.__name__}.access_regions dst count mismatch"
            for region_list, region_label in [(src_regions, "src"), (dst_regions, "dst")]:
                for ri, region in enumerate(region_list):
                    assert isinstance(region, tuple), \
                        f"{cls.__name__}.access_regions {region_label}[{ri}] must be tuple"
                    for d, dim_range in enumerate(region):
                        assert isinstance(dim_range, tuple) and len(dim_range) == 2, \
                            f"{cls.__name__}.access_regions {region_label}[{ri}][{d}] must be (start, end)"
                        start, end = dim_range
                        assert isinstance(start, int) and isinstance(end, int), \
                            f"{cls.__name__}.access_regions {region_label}[{ri}][{d}] bounds must be ints"
                        assert start < end, \
                            f"{cls.__name__}.access_regions {region_label}[{ri}][{d}] start >= end"
    except (RuntimeError, IndexError, ZeroDivisionError):
        pass

    # access_regions: dimensionality consistency and granularity match across all block indices

    if cls.test_cases:
        cls_args, input_args = cls.test_cases[0]
        test_inst = cls(*cls_args)
        args = test_inst.test_args(input_args)
        device = megakittens.schema.device.Device(type="cuda", index=0)
        test_src_metas = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                test_src_metas.append(megakittens.schema.tensor.TensorMeta(
                    dtype=megakittens.schema.dtype.DType.from_torch(arg.dtype),
                    shape=tuple(arg.shape),
                    device=device,
                ))
            elif isinstance(arg, Iterable):
                for t in arg:
                    if isinstance(t, torch.Tensor):
                        test_src_metas.append(megakittens.schema.tensor.TensorMeta(
                            dtype=megakittens.schema.dtype.DType.from_torch(t.dtype),
                            shape=tuple(t.shape),
                            device=device,
                        ))
        ref_result = test_inst.test_fn(*args)
        if isinstance(ref_result, torch.Tensor):
            test_dst_metas = (megakittens.schema.tensor.TensorMeta(
                dtype=megakittens.schema.dtype.DType.from_torch(ref_result.dtype),
                shape=tuple(ref_result.shape),
                device=device,
            ),)
        elif isinstance(ref_result, Iterable):
            test_dst_metas = tuple(
                megakittens.schema.tensor.TensorMeta(
                    dtype=megakittens.schema.dtype.DType.from_torch(t.dtype),
                    shape=tuple(t.shape),
                    device=device,
                ) for t in ref_result if isinstance(t, torch.Tensor)
            )
        else:
            test_dst_metas = ()

        assert len(test_src_metas) == len(test_inst.inputs), \
            f"{cls.__name__}: test_args produced {len(test_src_metas)} tensor args, expected {len(test_inst.inputs)} inputs"
        assert len(test_dst_metas) == len(test_inst.outputs), \
            f"{cls.__name__}: test_fn produced {len(test_dst_metas)} tensor outputs, expected {len(test_inst.outputs)} outputs"

        block_idx_list = test_inst.block_indices(test_src_metas, test_dst_metas)
        min_src_ndims = [len(spec.granularity) for spec in test_inst.inputs]
        min_dst_ndims = [len(spec.granularity) for spec in test_inst.outputs]
        first_src, first_dst = test_inst.access_regions(block_idx_list[0], test_src_metas, test_dst_metas)
        ref_src_ndims = [len(r) for r in first_src]
        ref_dst_ndims = [len(r) for r in first_dst]

        for bi, block_index in enumerate(block_idx_list):
            src_regions, dst_regions = test_inst.access_regions(block_index, test_src_metas, test_dst_metas)

            assert len(src_regions) == len(test_inst.inputs), \
                f"{cls.__name__} block {bi}: src_regions count {len(src_regions)} != inputs count {len(test_inst.inputs)}"
            assert len(dst_regions) == len(test_inst.outputs), \
                f"{cls.__name__} block {bi}: dst_regions count {len(dst_regions)} != outputs count {len(test_inst.outputs)}"

            for ri, (region, spec) in enumerate(zip(src_regions, test_inst.inputs)):
                assert len(region) >= min_src_ndims[ri], \
                    f"{cls.__name__} block {bi}: src_regions[{ri}] has {len(region)} dims, " \
                    f"fewer than inputs[{ri}].granularity ndim {min_src_ndims[ri]}"
                assert len(region) == ref_src_ndims[ri], \
                    f"{cls.__name__} block {bi}: src_regions[{ri}] has {len(region)} dims, " \
                    f"expected {ref_src_ndims[ri]} (consistent with block 0)"
                offset = len(region) - len(spec.granularity)
                for d, gran in enumerate(spec.granularity):
                    start, end = region[offset + d]
                    assert start % gran == 0, \
                        f"{cls.__name__} block {bi}: src_regions[{ri}] dim {offset + d} start {start} not divisible by granularity {gran}"
                    assert (end - start) % gran == 0, \
                        f"{cls.__name__} block {bi}: src_regions[{ri}] dim {offset + d} size {end - start} not divisible by granularity {gran}"
            for ri, (region, spec) in enumerate(zip(dst_regions, test_inst.outputs)):
                assert len(region) >= min_dst_ndims[ri], \
                    f"{cls.__name__} block {bi}: dst_regions[{ri}] has {len(region)} dims, " \
                    f"fewer than outputs[{ri}].granularity ndim {min_dst_ndims[ri]}"
                assert len(region) == ref_dst_ndims[ri], \
                    f"{cls.__name__} block {bi}: dst_regions[{ri}] has {len(region)} dims, " \
                    f"expected {ref_dst_ndims[ri]} (consistent with block 0)"
                offset = len(region) - len(spec.granularity)
                for d, gran in enumerate(spec.granularity):
                    start, end = region[offset + d]
                    assert start % gran == 0, \
                        f"{cls.__name__} block {bi}: dst_regions[{ri}] dim {offset + d} start {start} not divisible by granularity {gran}"
                    assert (end - start) % gran == 0, \
                        f"{cls.__name__} block {bi}: dst_regions[{ri}] dim {offset + d} size {end - start} not divisible by granularity {gran}"

    # validate() returns None

    try:
        result = inst.validate(src_metas, dst_metas)
        assert result is None, f"{cls.__name__}.validate must return None"
    except RuntimeError:
        pass

    # validate() rejects wrong input/output counts

    if inputs:
        wrong_src = tuple(
            megakittens.schema.tensor.TensorMeta(
                dtype=s.dtype, shape=s.granularity,
                device=megakittens.schema.device.Device(type="cuda", index=0),
            ) for s in inputs[:-1]
        )
        try:
            inst.validate(wrong_src, dst_metas)
            assert False, f"{cls.__name__}.validate should reject wrong input count"
        except (RuntimeError, IndexError):
            pass
    if outputs:
        wrong_dst = tuple(
            megakittens.schema.tensor.TensorMeta(
                dtype=s.dtype, shape=s.granularity,
                device=megakittens.schema.device.Device(type="cuda", index=0),
            ) for s in outputs[:-1]
        )
        try:
            inst.validate(src_metas, wrong_dst)
            assert False, f"{cls.__name__}.validate should reject wrong output count"
        except (RuntimeError, IndexError):
            pass

    # Custom op exists and signature matches test_args

    custom_op_name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cls.__name__).lower()
    mk_op = getattr(torch.ops.megakittens, custom_op_name, None)
    assert mk_op is not None, f"{cls.__name__} missing custom op torch.ops.megakittens.{custom_op_name}"
    schema = mk_op.default._schema
    schema_arg_types = [str(a.type) for a in schema.arguments]
    SCHEMA_TYPE_MAP = {
        "Tensor": torch.Tensor,
        "List[Tensor]": list,
        "str": str,
        "bool": bool,
        "int": int,
        "float": float,
    }
    if cls.test_cases:
        cls_args, input_args = cls.test_cases[0]
        try:
            test_inst = cls(*cls_args)
        except Exception as e:
            assert False, f"{cls.__name__}: cls(*{cls_args}) raised {e}"
        args = test_inst.test_args(input_args)
        assert len(args) == len(schema_arg_types), \
            f"{cls.__name__}.test_args returns {len(args)} args but custom op expects {len(schema_arg_types)}"
        for i, (arg, schema_type_str) in enumerate(zip(args, schema_arg_types)):
            expected_type = SCHEMA_TYPE_MAP.get(schema_type_str)
            if expected_type is not None:
                assert isinstance(arg, expected_type), \
                    f"{cls.__name__}.test_args()[{i}] is {type(arg).__name__} but custom op expects {schema_type_str}"

    # test_args works for all test_cases and bench_cases

    for label, cases in [("test_cases", cls.test_cases), ("bench_cases", cls.bench_cases)]:
        for j, (cls_args, input_args) in enumerate(cases):
            try:
                case_inst = cls(*cls_args)
            except Exception as e:
                assert False, f"{cls.__name__}.{label}[{j}]: cls(*{cls_args}) raised {e}"
            try:
                args = case_inst.test_args(input_args)
            except Exception as e:
                assert False, f"{cls.__name__}.{label}[{j}]: test_args({input_args}) raised {e}"
            assert isinstance(args, tuple), \
                f"{cls.__name__}.{label}[{j}]: test_args must return tuple"

    # __repr__, __eq__, __hash__

    r = repr(inst)
    assert isinstance(r, str) and cls.__name__ in r, f"{cls.__name__}.__repr__"
    assert inst == inst
    other = cls(*cls.test_cases[0][0]) if cls.test_cases else cls()
    if inst == other:
        assert hash(inst) == hash(other), f"{cls.__name__}: equal instances must have equal hashes"


try:
    # Support both standalone & pytest
    import pytest
    @pytest.mark.parametrize(
        "itype",
        collect_test_cases(),
        ids=[repr(itype) for itype in collect_test_cases()],
    )
    def test_itype_validity(itype):
        check_validity(itype)
except ImportError:
    pass


if __name__ == "__main__":
    names = sys.argv[1:] or None
    for itype in collect_test_cases(names):
        check_validity(itype)
        print(f"  PASS {itype!r}")
