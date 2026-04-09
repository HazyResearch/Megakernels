import megakittens
from .common import check


def collect_test_cases():
    test_cases = []
    for cls in megakittens.schema.itype.IType.__subclasses__():
        if "test_shapes" not in cls.__dict__:
            raise RuntimeError(f"{cls.__name__} must define test_shapes")
        if not isinstance(cls.__dict__["test_shapes"], list):
            raise RuntimeError(f"{cls.__name__}.test_shapes must be a list, got {type(cls.__dict__['test_shapes']).__name__}")
        if not all(isinstance(s, tuple) for s in cls.__dict__["test_shapes"]):
            raise RuntimeError(f"{cls.__name__}.test_shapes entries must be tuples")
        if "test_fn" not in cls.__dict__:
            raise RuntimeError(f"{cls.__name__} has test_shapes but no test_fn")
        if not callable(cls.__dict__["test_fn"]):
            raise RuntimeError(f"{cls.__name__}.test_fn must be callable")
        if "test_args" not in cls.__dict__:
            raise RuntimeError(f"{cls.__name__} has test_shapes but no test_args")
        if not callable(cls.__dict__["test_args"]):
            raise RuntimeError(f"{cls.__name__}.test_args must be callable")
        itype = cls()
        for shape in itype.test_shapes:
            test_cases.append((itype, shape))
    return test_cases


try:
    # Support both standalone & pytest
    import pytest
    @pytest.mark.parametrize(
        "itype, shape",
        collect_test_cases(),
        ids=[f"{itype.name}-{shape}" for itype, shape in collect_test_cases()],
    )
    def test_itype(itype, shape):
        check(itype.test_fn, itype.test_args(shape), atol=itype.test_atol, rtol=itype.test_rtol)
except ImportError:
    pass


if __name__ == "__main__":
    for itype, shape in collect_test_cases():
        max_diff, mean_diff = check(itype.test_fn, itype.test_args(shape), atol=itype.test_atol, rtol=itype.test_rtol)
        print(f"  PASS {itype.name} {shape} | max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
