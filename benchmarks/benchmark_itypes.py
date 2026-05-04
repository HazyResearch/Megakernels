import sys

import megakittens
from .common import benchmark


def collect_benchmark_cases(names=None):
    bench_cases = []
    for cls in megakittens.schema.itype.IType.__subclasses__():
        if "bench_cases" not in cls.__dict__:
            raise RuntimeError(f"{cls.__name__} must define bench_cases")
        if not isinstance(cls.__dict__["bench_cases"], list):
            raise RuntimeError(f"{cls.__name__}.bench_cases must be a list, got {type(cls.__dict__['bench_cases']).__name__}")
        if not all(isinstance(s, tuple) for s in cls.__dict__["bench_cases"]):
            raise RuntimeError(f"{cls.__name__}.bench_cases entries must be tuples")
        if "test_fn" not in cls.__dict__:
            raise RuntimeError(f"{cls.__name__} has bench_cases but no test_fn")
        if not callable(cls.__dict__["test_fn"]):
            raise RuntimeError(f"{cls.__name__}.test_fn must be callable")
        if "test_args" not in cls.__dict__:
            raise RuntimeError(f"{cls.__name__} has bench_cases but no test_args")
        if not callable(cls.__dict__["test_args"]):
            raise RuntimeError(f"{cls.__name__}.test_args must be callable")
        for cls_args, input_args in cls.bench_cases:
            itype = cls(*cls_args)
            if names and itype.name not in names:
                continue
            bench_cases.append((itype, input_args))
    return bench_cases


def benchmark_one(itype, bench_cases):
    has_flops = hasattr(itype, "bench_flops")
    has_bytes = hasattr(itype, "bench_bytes")

    header = f"{'shape':>28}  {'MK (us)':>10}  {'PT (us)':>10}"
    if has_flops:
        header += f"  {'MK TF':>8}  {'PT TF':>8}"
    if has_bytes:
        header += f"  {'MK GB/s':>10}  {'PT GB/s':>10}"
    header += f"  {'ratio':>7}"

    print(f"\n{itype!r} (bf16)")
    print(header)
    print("-" * len(header))

    for case in bench_cases:
        mk_ms, pt_ms = benchmark(itype.test_fn, itype.test_args(case), cluster_size=itype.cluster_size)
        line = f"{str(case):>28}  {mk_ms*1000:>10.1f}  {pt_ms*1000:>10.1f}"
        if has_flops:
            flops = itype.bench_flops(case)
            line += f"  {flops / mk_ms / 1e9:>8.1f}  {flops / pt_ms / 1e9:>8.1f}"
        if has_bytes:
            bytes_moved = itype.bench_bytes(case)
            line += f"  {bytes_moved / mk_ms / 1e6:>10.1f}  {bytes_moved / pt_ms / 1e6:>10.1f}"
        line += f"  {pt_ms/mk_ms:>6.2f}x"
        print(line)


if __name__ == "__main__":
    names = sys.argv[1:] or None
    grouped_bench_cases: dict = {}
    for itype, case in collect_benchmark_cases(names):
        grouped_bench_cases.setdefault(itype, []).append(case)
    for itype, bench_cases in grouped_bench_cases.items():
        benchmark_one(itype, bench_cases)
