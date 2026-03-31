import torch

from .common import check


def rms_upgate_silu(x: torch.Tensor, norm_weight: torch.Tensor, up_weights: torch.Tensor, gate_weights: torch.Tensor) -> torch.Tensor:
    return torch.ops.megakittens.rms_upgate_silu(x, norm_weight, up_weights, gate_weights, 1e-5)


def test_rms_upgate_silu() -> None:
    for M, N, I, atol in [
        (1, 2048, 256, 1e-1),
        (4, 2048, 512, 1e-1),
        (1, 2048, 8192, 2.0),   # Llama 1B: bf16 dot over N=2048 accumulates error
    ]:
        x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        nw = torch.randn(N, dtype=torch.bfloat16, device="cuda")
        up = torch.randn(I, N, dtype=torch.bfloat16, device="cuda")
        gate = torch.randn(I, N, dtype=torch.bfloat16, device="cuda")
        check(rms_upgate_silu, (x, nw, up, gate), atol=atol, rtol=1e-1)


if __name__ == "__main__":
    test_rms_upgate_silu()
