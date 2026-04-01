import torch
from .common import benchmark


def rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.ops.megakittens.rmsnorm(x, weight, 1e-6)


def benchmark_rmsnorm() -> None:
    print("RMSNorm (bf16)")
    print(f"{'shape':>20}  {'MK (us)':>10}  {'PT (us)':>10}  {'MK GB/s':>10}  {'PT GB/s':>10}  {'ratio':>7}")
    print("-" * 78)

    for M, N in [
        (32768, 256),
        (32768, 512),
        (32768, 1536),
        (32768, 2048),
        (32768, 4096),
        (32768, 8192),
        (32768, 16384),
    ]:
        x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, dtype=torch.bfloat16, device="cuda")

        mk_ms, pt_ms = benchmark(rmsnorm, (x, w))

        bytes_moved = M * N * 2 * 2 + N * 2
        mk_gbps = bytes_moved / mk_ms / 1e6
        pt_gbps = bytes_moved / pt_ms / 1e6

        print(f"  ({M:>5}, {N:>5})  {mk_ms*1000:>10.2f}  {pt_ms*1000:>10.2f}  {mk_gbps:>10.1f}  {pt_gbps:>10.1f}  {pt_ms/mk_ms:>6.2f}x")


if __name__ == "__main__":
    benchmark_rmsnorm()
