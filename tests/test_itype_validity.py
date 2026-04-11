import sys

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
