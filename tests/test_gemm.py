import torch
from .common import check


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def test_gemm() -> None:
    for M, N, K in [(256, 256, 64), (256, 256, 256), (512, 512, 256), (1024, 1024, 512), (2560, 2560, 64)]:
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        check(gemm, (a, b))


if __name__ == "__main__":
    test_gemm()
