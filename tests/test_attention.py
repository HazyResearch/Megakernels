import torch
import torch.nn.functional as F
from .common import check


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.ops.megakittens.attention(q, k, v)


def test_attention() -> None:
    head_dim = 128
    for batch, seq_len, num_heads in [
        (16, 1024, 16),
        (16, 2048, 16),
        (16, 4096, 16),
        (16, 8192, 16),
        (16, 16384, 16),
    ]:
        # BSHD layout: (batch, seq_len, num_heads, head_dim)
        q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        max_diff, mean_diff = check(attention, (q, k, v), atol=1e-2, rtol=1e-2)
        print(f"  b={batch} s={seq_len} h={num_heads} d={head_dim} | max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")


if __name__ == "__main__":
    test_attention()
