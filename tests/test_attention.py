import torch
import torch.nn.functional as F
from .common import check


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.ops.megakittens.attention(q, k, v)


def test_attention() -> None:
    head_dim = 128
    for batch, seq_len, num_heads in [
        (1, 512, 1),
        (1, 1024, 1),
        (1, 2048, 1),
        (2, 512, 4),
        (2, 1024, 8),
    ]:
        # BSHD layout: (batch, seq_len, num_heads, head_dim)
        q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        check(attention, (q, k, v), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    test_attention()
