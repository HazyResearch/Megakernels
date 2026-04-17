import torch
from .common import benchmark


NUM_LAYERS = 3


def mlp(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    for i in range(NUM_LAYERS):
        x = torch.relu(x @ W[i] + b[i])
    return x


def benchmark_mlp() -> None:
    print(f"{NUM_LAYERS}-layer MLP (bf16)")
    print(f"{'(M, H)':>15}  {'MK (us)':>10}  {'PT (us)':>10}  {'ratio':>7}")
    print("-" * 50)

    for M, H in [
        (4096, 4096),
        (8192, 4096),
        (16384, 4096),
        (16384, 8192),
    ]:
        x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        W = torch.randn(NUM_LAYERS, H, H, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(NUM_LAYERS, M, H, dtype=torch.bfloat16, device="cuda")

        mk_ms, pt_ms = benchmark(mlp, (x, W, b))
        print(f"  ({M:>5},{H:>5})  {mk_ms*1000:>10.2f}  {pt_ms*1000:>10.2f}  {pt_ms/mk_ms:>6.2f}x")


if __name__ == "__main__":
    benchmark_mlp()
