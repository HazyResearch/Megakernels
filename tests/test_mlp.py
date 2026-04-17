import torch

from .common import check


NUM_LAYERS = 3


def mlp(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    for i in range(NUM_LAYERS):
        x = torch.relu(x @ W[i] + b[i])
    return x


def mlp_sliced(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    W = W[NUM_LAYERS:]
    b = b[NUM_LAYERS:]
    for i in range(NUM_LAYERS):
        x = torch.relu(x @ W[i] + b[i])
    return x


def mlp_strided(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    W = W[::2]
    b = b[::2]
    for i in range(NUM_LAYERS):
        x = torch.relu(x @ W[i] + b[i])
    return x


def test_mlp() -> None:
    for M, H in [(512, 512), (1024, 512), (1024, 1024)]:
        x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        W = torch.randn(NUM_LAYERS, H, H, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(NUM_LAYERS, M, H, dtype=torch.bfloat16, device="cuda")
        check(mlp, (x, W, b))


def test_mlp_sliced() -> None:
    for M, H in [(512, 512), (1024, 512), (1024, 1024)]:
        x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        W = torch.randn(2 * NUM_LAYERS, H, H, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(2 * NUM_LAYERS, M, H, dtype=torch.bfloat16, device="cuda")
        check(mlp_sliced, (x, W, b))


def test_mlp_strided_unsupported() -> None:
    M, H = 512, 512
    x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(2 * NUM_LAYERS, H, H, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(2 * NUM_LAYERS, M, H, dtype=torch.bfloat16, device="cuda")
    try:
        check(mlp_strided, (x, W, b))
    except RuntimeError as e:
        assert "step != 1 is not supported" in str(e), f"Unexpected error: {e}"
    else:
        raise AssertionError("Expected RuntimeError for non-1 stride")


if __name__ == "__main__":
    test_mlp()
    test_mlp_sliced()
    test_mlp_strided_unsupported()
