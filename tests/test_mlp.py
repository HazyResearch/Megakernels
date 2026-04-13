import torch

from .common import check


def mlp(
    x: torch.Tensor, 
    W1: torch.Tensor, b1: torch.Tensor,
    W2: torch.Tensor, b2: torch.Tensor,
    W3: torch.Tensor, b3: torch.Tensor
) -> torch.Tensor:
    x = torch.relu(x @ W1 + b1)
    x = torch.relu(x @ W2 + b2)
    x = torch.relu(x @ W3 + b3)
    return x


def test_mlp() -> None:
    for M, H in [(512, 512), (1024, 512), (1024, 1024)]:
        x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        W1 = torch.randn(H, H, dtype=torch.bfloat16, device="cuda")
        b1 = torch.randn(H, dtype=torch.bfloat16, device="cuda").unsqueeze(0).expand(M, H).contiguous()
        W2 = torch.randn(H, H, dtype=torch.bfloat16, device="cuda")
        b2 = torch.randn(H, dtype=torch.bfloat16, device="cuda").unsqueeze(0).expand(M, H).contiguous()
        W3 = torch.randn(H, H, dtype=torch.bfloat16, device="cuda")
        b3 = torch.randn(H, dtype=torch.bfloat16, device="cuda").unsqueeze(0).expand(M, H).contiguous()
        check(mlp, (x, W1, b1, W2, b2, W3, b3))


if __name__ == "__main__":
    test_mlp()
