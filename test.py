import torch
torch.manual_seed(42)

import megakittens


@megakittens.compile(debug=True, save_dag=True, save_schedule=True)
def mlp(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    z = torch.matmul(x, W) + b
    return torch.relu(z)


if __name__ == "__main__":
    M, N, K = 4, 8, 16
    x = torch.rand(M, N, dtype=torch.bfloat16, device="cuda")
    W = torch.rand(N, K, dtype=torch.bfloat16, device="cuda")
    b = torch.rand(K, dtype=torch.bfloat16, device="cuda")

    print(f"M={M} N={N} K={K}")
    print(mlp(x, W, b))
