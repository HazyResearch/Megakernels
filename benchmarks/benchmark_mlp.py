import torch
from .common import benchmark


NUM_LAYERS = 3


def mlp(
    x: torch.Tensor, 
    Ws: list[torch.Tensor],
    bs: list[torch.Tensor],
) -> torch.Tensor:
    for i in range(NUM_LAYERS):
        x = torch.relu(x @ Ws[i] + bs[i])
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
        args = (torch.randn(M, H, dtype=torch.bfloat16, device="cuda"), [], [])
        for _ in range(NUM_LAYERS):
            args[1].append(torch.randn(H, H, dtype=torch.bfloat16, device="cuda"))
            args[2].append(torch.randn(H, dtype=torch.bfloat16, device="cuda").unsqueeze(0).expand(M, H).contiguous())

        mk_ms, pt_ms = benchmark(mlp, args)

        print(f"  ({M:>5},{H:>5})  {mk_ms*1000:>10.2f}  {pt_ms*1000:>10.2f}  {pt_ms/mk_ms:>6.2f}x")


if __name__ == "__main__":
    benchmark_mlp()
