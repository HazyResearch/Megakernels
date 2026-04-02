import torch

from .common import benchmark


B300_BW_BYTES_PER_SEC = 8_000_000_000_000


def proj_residual(
    x: torch.Tensor,
    weights: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.megakittens.matvec_adds(x, weights, residual)


def proj_residual_unfused(
    x: torch.Tensor,
    weights: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    return residual + x @ weights.T


def _roofline_us(M: int, I: int, N: int) -> float:
    read_bytes = (M * I + N * I + M * N) * 2
    write_bytes = M * N * 2
    total_bytes = read_bytes + write_bytes
    secs = total_bytes / B300_BW_BYTES_PER_SEC
    return secs * 1e6


def _bench_unfused(
    x: torch.Tensor,
    weights: torch.Tensor,
    residual: torch.Tensor,
    warmup: int = 500,
    iters: int = 100,
) -> float:
    for _ in range(warmup):
        proj_residual_unfused(x, weights, residual)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        proj_residual_unfused(x, weights, residual)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000


M_LABELS = {1: "decode", 32: "prefill-32", 128: "prefill-128"}


def _bench_section(title: str, I: int, N: int, batches: tuple) -> None:
    print(title)
    print()
    print(
        f"{'M':>5}  {'use case':>12}  {'(M, I, N)':>24}  {'MK (us)':>10}  {'PT fused (us)':>14}  "
        f"{'unfused (us)':>13}  {'roofline (us)':>14}  {'MK/roof':>8}"
    )
    print("-" * 115)

    for M in batches:
        x = torch.randn(M, I, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, I, dtype=torch.bfloat16, device="cuda")
        res = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

        mk_ms, pt_ms = benchmark(proj_residual, (x, w, res))
        mk_us = mk_ms * 1000
        pt_us = pt_ms * 1000
        unfused_us = _bench_unfused(x, w, res)
        roof_us = _roofline_us(M, I, N)

        shape = f"({M}, {I}, {N})"
        use = M_LABELS.get(M, f"M={M}")
        print(
            f"{M:>5}  {use:>12}  {shape:>24}  {mk_us:>10.2f}  {pt_us:>14.2f}  "
            f"{unfused_us:>13.2f}  {roof_us:>14.2f}  {mk_us / roof_us:>7.1f}x"
        )


def benchmark_proj_residual() -> None:
    print("Llama 3.2 1B proj_residual benchmark (bf16)")
    print(f"B300 theoretical peak bandwidth: {B300_BW_BYTES_PER_SEC / 1e12:.0f} TB/s")
    print()

    batches = (1, 32, 128)

    _bench_section(
        "Instruction 4: O-proj + residual  —  attn_out(M,2048) @ o_proj(2048,2048) + residual(M,2048)",
        I=2048, N=2048, batches=batches,
    )
    print()
    _bench_section(
        "Instruction 6: Down-proj + residual  —  mlp_out(M,8192) @ down_proj(2048,8192) + residual(M,2048)",
        I=8192, N=2048, batches=batches,
    )


if __name__ == "__main__":
    benchmark_proj_residual()
