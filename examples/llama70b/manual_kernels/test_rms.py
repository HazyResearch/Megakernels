from __future__ import annotations

import torch

from . import _C


HIDDEN_DIM = 8192
EPS_VAL = 1e-5


def _ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    return (x_f * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


def main():
    torch.manual_seed(0)
    for B in (128, 512, 1024, 2048):
        x = torch.randn(B, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device="cuda")
        eps = torch.tensor([EPS_VAL], dtype=torch.float32, device="cuda")
        out = torch.empty_like(x)

        _C.rms_forward(x, weight, eps, out)
        torch.cuda.synchronize()

        ref = _ref_rmsnorm(x, weight, EPS_VAL)
        diff = (out.float() - ref.float()).abs()

        warmup, iters = 20, 200
        for _ in range(warmup):
            _C.rms_forward(x, weight, eps, out)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _C.rms_forward(x, weight, eps, out)
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000.0 / iters

        print(f"B={B:5d}  max_err={diff.max().item():.4g}  mean_err={diff.mean().item():.4g}  {us:7.2f} us")


if __name__ == "__main__":
    main()
