import torch
from .common import benchmark

def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def benchmark_gemm() -> None:
    print("Gemm (bf16)")
    print(f"{'shape':>20}  {'MK (us)':>10}  {'PT (us)':>10}  {'MK TF':>8}  {'PT TF':>8}  {'ratio':>7}")
    print("-" * 68)

    for M, N, K in [
        (16384, 16384, 16384),
        (16384, 32768, 16384),
        (32768, 16384, 16384),
        (32768, 32768, 16384),
    ]:
        a = torch.rand(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.rand(K, N, dtype=torch.bfloat16, device="cuda")

        mk_ms, pt_ms = benchmark(gemm, (a, b))

        flops = 2.0 * M * N * K
        mk_tf = flops / mk_ms / 1e9
        pt_tf = flops / pt_ms / 1e9

        print(f"  ({M:>5},{N:>5},{K:>5})  {mk_ms*1000:>10.1f}  {pt_ms*1000:>10.1f}  {mk_tf:>8.1f}  {pt_tf:>8.1f}  {pt_ms/mk_ms:>6.2f}x")


if __name__ == "__main__":
    benchmark_gemm()
