import torch
torch.manual_seed(42)

import megakittens
from .common import check


@torch.library.custom_op("megakittens::rms_norm", mutates_args=())
def rms_norm_op(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rms_norm(x, [x.shape[-1]], weight, eps)

@rms_norm_op.register_fake
def _rms_norm_fake(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)


def rms_norm(x, weight):
    return torch.ops.megakittens.rms_norm(x, weight, 1e-6)


def test_rmsnorm():
    for M, N in [
        (1, 2048),
        (4, 2048),
        (32, 2048),
        (16, 4096),
        (32, 4096),
        (8, 8192),
    ]:
        x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, dtype=torch.bfloat16, device="cuda")
        check(rms_norm, (x, w), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    test_rmsnorm()
