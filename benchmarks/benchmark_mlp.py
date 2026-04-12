import torch
from .common import benchmark


def mlp(
    x: torch.Tensor, 
    W1: torch.Tensor, b1: torch.Tensor,
    W2: torch.Tensor, b2: torch.Tensor,
    W3: torch.Tensor, b3: torch.Tensor,
    W4: torch.Tensor, b4: torch.Tensor,
    W5: torch.Tensor, b5: torch.Tensor,
    W6: torch.Tensor, b6: torch.Tensor,
    W7: torch.Tensor, b7: torch.Tensor,
    W8: torch.Tensor, b8: torch.Tensor,
) -> torch.Tensor:
    x = torch.relu(x @ W1 + b1)
    x = torch.relu(x @ W2 + b2)
    x = torch.relu(x @ W3 + b3)
    x = torch.relu(x @ W4 + b4)
    x = torch.relu(x @ W5 + b5)
    x = torch.relu(x @ W6 + b6)
    x = torch.relu(x @ W7 + b7)
    x = torch.relu(x @ W8 + b8)
    return x


def benchmark_mlp() -> None:
    print("8-layer MLP (bf16)")
    print(f"{'(M, H)':>15}  {'MK (us)':>10}  {'PT (us)':>10}  {'ratio':>7}")
    print("-" * 50)

    for M, H in [
        (4096, 4096),
        (8192, 4096),
        (16384, 4096),
        (16384, 8192),
    ]:
        x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        args = [x]
        for _ in range(8):
            W = torch.randn(H, H, dtype=torch.bfloat16, device="cuda")
            b = torch.randn(H, dtype=torch.bfloat16, device="cuda").unsqueeze(0).expand(M, H).contiguous()
            args.extend([W, b])

        mk_ms, pt_ms = benchmark(mlp, tuple(args))

        print(f"  ({M:>5},{H:>5})  {mk_ms*1000:>10.2f}  {pt_ms*1000:>10.2f}  {pt_ms/mk_ms:>6.2f}x")


if __name__ == "__main__":
    benchmark_mlp()
