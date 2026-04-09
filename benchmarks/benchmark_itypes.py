import sys

import megakittens
from .common import benchmark


def collect_benchmark_cases(names=None):
    benchmark_cases = []
    for cls in megakittens.schema.itype.IType.__subclasses__():
        if "bench_shapes" not in cls.__dict__:
            raise RuntimeError(f"{cls.__name__} must define bench_shapes")
        if not isinstance(cls.__dict__["bench_shapes"], list):
            raise RuntimeError(f"{cls.__name__}.bench_shapes must be a list, got {type(cls.__dict__['bench_shapes']).__name__}")
        if not all(isinstance(s, tuple) for s in cls.__dict__["bench_shapes"]):
            raise RuntimeError(f"{cls.__name__}.bench_shapes entries must be tuples")
        if "test_fn" not in cls.__dict__:
            raise RuntimeError(f"{cls.__name__} has bench_shapes but no test_fn")
        if not callable(cls.__dict__["test_fn"]):
            raise RuntimeError(f"{cls.__name__}.test_fn must be callable")
        if "make_args" not in cls.__dict__:
            raise RuntimeError(f"{cls.__name__} has bench_shapes but no make_args")
        if not callable(cls.__dict__["make_args"]):
            raise RuntimeError(f"{cls.__name__}.make_args must be callable")
        itype = cls()
        if names and itype.name not in names:
            continue
        benchmark_cases.append(itype)
    return benchmark_cases


def benchmark_one(itype):
    has_flops = hasattr(itype, "bench_flops")
    has_bytes = hasattr(itype, "bench_bytes")

    print(f"\n{itype.name} (bf16)")
    if has_flops:
        print(f"{'shape':>28}  {'MK (us)':>10}  {'PT (us)':>10}  {'MK TF':>8}  {'PT TF':>8}  {'ratio':>7}")
    elif has_bytes:
        print(f"{'shape':>28}  {'MK (us)':>10}  {'PT (us)':>10}  {'MK GB/s':>10}  {'PT GB/s':>10}  {'ratio':>7}")
    else:
        print(f"{'shape':>28}  {'MK (us)':>10}  {'PT (us)':>10}  {'ratio':>7}")
    print("-" * 80)

    for shape in itype.bench_shapes:
        mk_ms, pt_ms = benchmark(itype.test_fn, itype.make_args(shape))
        if has_flops:
            flops = itype.bench_flops(shape)
            mk_tf = flops / mk_ms / 1e9
            pt_tf = flops / pt_ms / 1e9
            print(f"  {str(shape).rjust(26)}  {mk_ms*1000:>10.1f}  {pt_ms*1000:>10.1f}  {mk_tf:>8.1f}  {pt_tf:>8.1f}  {pt_ms/mk_ms:>6.2f}x")
        elif has_bytes:
            bytes_moved = itype.bench_bytes(shape)
            mk_gbps = bytes_moved / mk_ms / 1e6
            pt_gbps = bytes_moved / pt_ms / 1e6
            print(f"  {str(shape).rjust(26)}  {mk_ms*1000:>10.1f}  {pt_ms*1000:>10.1f}  {mk_gbps:>10.1f}  {pt_gbps:>10.1f}  {pt_ms/mk_ms:>6.2f}x")
        else:
            print(f"  {str(shape).rjust(26)}  {mk_ms*1000:>10.1f}  {pt_ms*1000:>10.1f}  {pt_ms/mk_ms:>6.2f}x")


if __name__ == "__main__":
    names = sys.argv[1:] or None
    for itype in collect_benchmark_cases(names):
        benchmark_one(itype)
