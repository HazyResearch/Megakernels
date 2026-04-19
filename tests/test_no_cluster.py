import sys

from .common import check
from .test_itypes import collect_test_cases


def collect_no_cluster_test_cases(names=None):
    cluster_agnostic_itypes = {"elementwise_unary", "elementwise_binary", "rmsnorm"}
    names = set(names) & cluster_agnostic_itypes if names else cluster_agnostic_itypes
    return collect_test_cases(names)


try:
    import pytest
    @pytest.mark.parametrize(
        "itype, case",
        collect_no_cluster_test_cases(),
        ids=[f"{itype!r}-{case}" for itype, case in collect_no_cluster_test_cases()],
    )
    def test_no_cluster(itype, case):
        check(itype.test_fn, itype.test_args(case), atol=itype.test_atol, rtol=itype.test_rtol, cluster_size=1)
except ImportError:
    pass


if __name__ == "__main__":
    names = sys.argv[1:] or None
    for itype, case in collect_no_cluster_test_cases(names):
        max_diff, mean_diff = check(itype.test_fn, itype.test_args(case), atol=itype.test_atol, rtol=itype.test_rtol, cluster_size=1)
        print(f"  PASS {itype.name} {case} | max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
