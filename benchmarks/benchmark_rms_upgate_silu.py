import torch

from .common import benchmark


B300_BW_BYTES_PER_SEC = 8_000_000_000_000


def rms_upgate_silu(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weights: torch.Tensor,
    gate_weights: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.megakittens.rms_upgate_silu(x, norm_weight, up_weights, gate_weights, 1e-5)


def rms_upgate_silu_unfused(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weights: torch.Tensor,
    gate_weights: torch.Tensor,
) -> torch.Tensor:
    h = torch.rms_norm(x, [x.shape[-1]], norm_weight, 1e-5)
    up = h @ up_weights.T
    gate = h @ gate_weights.T
    return up * torch.nn.functional.silu(gate)


def _roofline_us(M: int, N: int, I: int) -> float:
    read_bytes = (M * N + N + 2 * I * N) * 2
    write_bytes = M * I * 2
    total_bytes = read_bytes + write_bytes
    secs = total_bytes / B300_BW_BYTES_PER_SEC
    return secs * 1e6


def _bench_unfused(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weights: torch.Tensor,
    gate_weights: torch.Tensor,
    warmup: int = 500,
    iters: int = 100,
) -> float:
    for _ in range(warmup):
        rms_upgate_silu_unfused(x, norm_weight, up_weights, gate_weights)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        rms_upgate_silu_unfused(x, norm_weight, up_weights, gate_weights)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000


M_LABELS = {1: "decode", 32: "prefill-32", 128: "prefill-128"}


def benchmark_rms_upgate_silu() -> None:
    n_hidden = 2048
    inter = 8192
    batches = (1, 32, 128)

    print("Llama 3.2 1B rms_upgate_silu benchmark (bf16)")
    print("Instruction 5: RMSNorm + UpGate + SiLU  —  rms_norm(x(M,2048)) @ up/gate(8192,2048).T -> silu_out(M,8192)")
    print(f"B300 theoretical peak bandwidth: {B300_BW_BYTES_PER_SEC / 1e12:.0f} TB/s")
    print()
    print(
        f"{'M':>5}  {'use case':>12}  {'(M, N, I)':>24}  {'MK (us)':>10}  {'PT fused (us)':>14}  "
        f"{'unfused (us)':>13}  {'roofline (us)':>14}  {'MK/roof':>8}"
    )
    print("-" * 115)

    for M in batches:
        x = torch.randn(M, n_hidden, dtype=torch.bfloat16, device="cuda")
        nw = torch.randn(n_hidden, dtype=torch.bfloat16, device="cuda")
        up = torch.randn(inter, n_hidden, dtype=torch.bfloat16, device="cuda")
        gate = torch.randn(inter, n_hidden, dtype=torch.bfloat16, device="cuda")

        mk_ms, pt_ms = benchmark(rms_upgate_silu, (x, nw, up, gate))
        mk_us = mk_ms * 1000
        pt_us = pt_ms * 1000
        unfused_us = _bench_unfused(x, nw, up, gate)
        roof_us = _roofline_us(M, n_hidden, inter)

        shape = f"({M}, {n_hidden}, {inter})"
        use = M_LABELS.get(M, f"M={M}")
        print(
            f"{M:>5}  {use:>12}  {shape:>24}  {mk_us:>10.2f}  {pt_us:>14.2f}  "
            f"{unfused_us:>13.2f}  {roof_us:>14.2f}  {mk_us / roof_us:>7.1f}x"
        )


if __name__ == "__main__":
    benchmark_rms_upgate_silu()
