import torch
torch.manual_seed(42)

from .common import check


def add(a, b):
    return a + b


def test_add():
    for M, N in [(128, 256), (256, 512), (512, 1024), (1024, 2048)]:
        a = torch.rand(M, N, dtype=torch.bfloat16, device="cuda")
        b = torch.rand(M, N, dtype=torch.bfloat16, device="cuda")
        check(add, (a, b))


if __name__ == "__main__":
    test_add()
