import torch

from .common import check


def rms_lm_head(x: torch.Tensor, norm_weight: torch.Tensor, lm_head: torch.Tensor) -> torch.Tensor:
    return torch.ops.megakittens.rms_lm_head(x, norm_weight, lm_head, 1e-6)


def test_rms_lm_head() -> None:
    for M, N, V in [
        (128, 2048, 256),
        (128, 2048, 128),
        (256, 2048, 512),
    ]:
        x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        nw = torch.randn(N, dtype=torch.bfloat16, device="cuda")
        lh = torch.randn(V, N, dtype=torch.bfloat16, device="cuda")
        check(rms_lm_head, (x, nw, lh), atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    test_rms_lm_head()
