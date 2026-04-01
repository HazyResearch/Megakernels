import torch
from .common import check


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def test_relu() -> None:
    for M, N in [(128, 256), (256, 512), (512, 1024), (1024, 2048), (1280, 2048)]:
        x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        check(relu, (x,))


if __name__ == "__main__":
    test_relu()
