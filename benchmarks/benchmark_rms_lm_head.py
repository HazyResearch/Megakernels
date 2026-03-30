import torch

from .common import benchmark


B300_BW_BYTES_PER_SEC = 8_000_000_000_000


def rms_lm_head(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.megakittens.rms_lm_head(x, norm_weight, lm_head, 1e-6)


def rms_lm_head_unfused(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
) -> torch.Tensor:
    h = torch.rms_norm(x, [x.shape[-1]], norm_weight, 1e-6)
    return h @ lm_head.T


def _roofline_us(M: int, N: int, V: int) -> float:
    read_bytes = (M * N + N + V * N) * 2
    write_bytes = M * V * 2
    total_bytes = read_bytes + write_bytes
    secs = total_bytes / B300_BW_BYTES_PER_SEC
    return secs * 1e6


def _bench_unfused(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    warmup: int = 500,
    iters: int = 100,
) -> float:
    for _ in range(warmup):
        rms_lm_head_unfused(x, norm_weight, lm_head)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        rms_lm_head_unfused(x, norm_weight, lm_head)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000


def benchmark_rms_lm_head() -> None:
    n_hidden = 2048
    vocab = 128_256
    batches = (1, 32, 128)

    print("RMSNorm + LM head (bf16), Llama 1B-like (N=2048, V=128256)")
    print(f"B300 theoretical peak bandwidth: {B300_BW_BYTES_PER_SEC / 1e12:.0f} TB/s")
    print()
    print(
        f"{'(M, N, V)':>24}  {'MK (us)':>10}  {'PT fused (us)':>14}  "
        f"{'unfused (us)':>13}  {'roofline (us)':>14}  {'MK/roof':>8}"
    )
    print("-" * 96)

    for M in batches:
        x = torch.randn(M, n_hidden, dtype=torch.bfloat16, device="cuda")
        nw = torch.randn(n_hidden, dtype=torch.bfloat16, device="cuda")
        lh = torch.randn(vocab, n_hidden, dtype=torch.bfloat16, device="cuda")

        mk_ms, pt_ms = benchmark(rms_lm_head, (x, nw, lh))
        mk_us = mk_ms * 1000
        pt_us = pt_ms * 1000
        unfused_us = _bench_unfused(x, nw, lh)
        roof_us = _roofline_us(M, n_hidden, vocab)

        label = f"({M}, {n_hidden}, {vocab})"
        print(
            f"{label:>24}  {mk_us:>10.2f}  {pt_us:>14.2f}  "
            f"{unfused_us:>13.2f}  {roof_us:>14.2f}  {mk_us / roof_us:>7.1f}x"
        )


if __name__ == "__main__":
    benchmark_rms_lm_head()
