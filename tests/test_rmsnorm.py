import torch
from .common import check


def rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.ops.megakittens.rmsnorm(x, weight, 1e-6)


def test_rmsnorm() -> None:
    for M, N in [(1, 2048), (4, 2048), (32, 2048), (16, 4096), (32, 4096), (8, 8192)]:
        x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, dtype=torch.bfloat16, device="cuda")
        check(rmsnorm, (x, w), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    test_rmsnorm()
