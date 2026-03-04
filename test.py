import torch
torch.manual_seed(42)

import megakittens


@megakittens.compile(debug=True, save_graph=True)
def mlp(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    z = torch.matmul(x, W) + b
    return torch.relu(z)


if __name__ == "__main__":
    M, N, K = 4, 8, 16
    x = torch.rand(M, N)
    W = torch.rand(N, K)
    b = torch.rand(K)

    print(f"M={M} N={N} K={K}")
    print(mlp(x, W, b))
