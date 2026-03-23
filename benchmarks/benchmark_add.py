import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from benchmarks.common import benchmark


def add(a, b):
    return a + b


def benchmark_add():
    print("Add (bf16)")
    print(f"{'shape':>20}  {'MK (us)':>10}  {'PT (us)':>10}  {'MK GB/s':>10}  {'PT GB/s':>10}  {'ratio':>7}")
    print("-" * 78)

    for M, N in [
        (4096, 4096),
        (131072, 4096),
        (4096, 131072),
        (16384, 16384),
        (131072, 131072),
    ]:
        a = torch.rand(M, N, dtype=torch.bfloat16, device="cuda")
        b = torch.rand(M, N, dtype=torch.bfloat16, device="cuda")

        mk_ms, pt_ms = benchmark(add, (a, b))

        bytes_moved = M * N * 2 * 3  # 2 bytes/bf16, 3 tensors (2 read + 1 write)
        mk_gbps = bytes_moved / mk_ms / 1e6
        pt_gbps = bytes_moved / pt_ms / 1e6

        print(f"  ({M:>5}, {N:>5})  {mk_ms*1000:>10.2f}  {pt_ms*1000:>10.2f}  {mk_gbps:>10.1f}  {pt_gbps:>10.1f}  {pt_ms/mk_ms:>6.2f}x")


if __name__ == "__main__":
    benchmark_add()
