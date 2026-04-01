import torch
from .common import check


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


def test_add() -> None:
    for M, N in [(128, 256), (256, 512), (512, 1024), (1024, 2048), (1280, 2048)]:
        a = torch.rand(M, N, dtype=torch.bfloat16, device="cuda")
        b = torch.rand(M, N, dtype=torch.bfloat16, device="cuda")
        check(add, (a, b))


if __name__ == "__main__":
    test_add()
