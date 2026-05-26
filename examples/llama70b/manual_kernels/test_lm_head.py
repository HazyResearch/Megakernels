from __future__ import annotations

try:
    import pytest
except ImportError:
    pytest = None

import torch

from . import _C


HIDDEN_DIM = 8192
VOCAB_SIZE = 128256


def _ref(hidden: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (hidden.float() @ w[0].float().transpose(-1, -2)).to(hidden.dtype)


def _make_case(M: int, N: int, K: int, device: str = "cuda"):
    hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(1, N, K, dtype=torch.bfloat16, device=device) * (K ** -0.5)
    return hidden, w


_CASES = [
    (512,  256,        HIDDEN_DIM),   # smallest valid: M=M_INST, N=Nb
    (512,  VOCAB_SIZE, HIDDEN_DIM),
    (1024, VOCAB_SIZE, HIDDEN_DIM),
    (2048, VOCAB_SIZE, HIDDEN_DIM),
]


if pytest is not None:
    @pytest.mark.parametrize("M,N,K", _CASES)
    def test_lm_head(M, N, K):
        torch.manual_seed(0)
        hidden, w = _make_case(M, N, K)
        ref = _ref(hidden, w)
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        _C.lm_head_forward(hidden, w, out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def main():
    torch.manual_seed(0)
    for M, N, K in _CASES:
        hidden, w = _make_case(M, N, K)
        ref = _ref(hidden, w)
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        _C.lm_head_forward(hidden, w, out)
        torch.cuda.synchronize()
        err = (out.float() - ref.float()).abs()

        warmup, iters = 20, 200
        for _ in range(warmup):
            _C.lm_head_forward(hidden, w, out)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _C.lm_head_forward(hidden, w, out)
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000.0 / iters
        tflops = 2.0 * M * N * K / (us * 1e-6) / 1e12
        print(f"M={M:5d} N={N:6d} K={K:5d}  err={err.max().item():.4g}  "
              f"{us:7.2f} us  {tflops:6.1f} TFLOPS")


if __name__ == "__main__":
    main()
