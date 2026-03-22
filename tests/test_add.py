import torch
torch.manual_seed(42)

from .common import check


def add(a, b):
    return a + b


def test_add():
    a = torch.rand(128, 256, dtype=torch.bfloat16, device="cuda")
    b = torch.rand(128, 256, dtype=torch.bfloat16, device="cuda")
    check(add, (a, b))


def test_add_large():
    a = torch.rand(256, 512, dtype=torch.bfloat16, device="cuda")
    b = torch.rand(256, 512, dtype=torch.bfloat16, device="cuda")
    check(add, (a, b))


if __name__ == "__main__":
    test_add()
    test_add_large()
