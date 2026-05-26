from __future__ import annotations

try:
    import pytest
except ImportError:
    pytest = None

import torch

from . import _C


HIDDEN_DIM = 8192


def _ref(hidden: torch.Tensor, attn_out: torch.Tensor, o_w: torch.Tensor) -> torch.Tensor:
    update = (attn_out.float() @ o_w[0].float().transpose(-1, -2)).to(hidden.dtype)
    return hidden + update


def _make_case(M: int, N: int, K: int, device: str = "cuda"):
    hidden = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    attn_out = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    o_w = torch.randn(1, N, K, dtype=torch.bfloat16, device=device) * (K ** -0.5)
    return hidden, attn_out, o_w


_CASES = [
    (512,  256,       HIDDEN_DIM),
    (512,  HIDDEN_DIM, HIDDEN_DIM),
    (1024, HIDDEN_DIM, HIDDEN_DIM),
    (1024, HIDDEN_DIM, 28672),
    (2048, HIDDEN_DIM, HIDDEN_DIM),
]


if pytest is not None:
    @pytest.mark.parametrize("M,N,K", _CASES)
    def test_o_proj_residual(M, N, K):
        torch.manual_seed(0)
        hidden, attn_out, o_w = _make_case(M, N, K)
        ref = _ref(hidden, attn_out, o_w)
        _C.o_proj_residual_forward(attn_out, o_w, hidden)
        torch.cuda.synchronize()
        torch.testing.assert_close(hidden, ref, atol=1e-2, rtol=1e-2)


def main():
    torch.manual_seed(0)
    for M, N, K in _CASES:
        hidden, attn_out, o_w = _make_case(M, N, K)
        hidden_ref = _ref(hidden, attn_out, o_w)
        hidden_test = hidden.clone()
        _C.o_proj_residual_forward(attn_out, o_w, hidden_test)
        torch.cuda.synchronize()
        err = (hidden_test.float() - hidden_ref.float()).abs()

        # Use the original hidden each iteration so the test value doesn't drift.
        warmup, iters = 20, 200
        for _ in range(warmup):
            hidden_bench = hidden.clone()
            _C.o_proj_residual_forward(attn_out, o_w, hidden_bench)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _C.o_proj_residual_forward(attn_out, o_w, hidden_bench)
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000.0 / iters
        tflops = 2.0 * M * N * K / (us * 1e-6) / 1e12
        print(f"M={M:5d} N={N:6d} K={K:5d}  err={err.max().item():.4g}  "
              f"{us:7.2f} us  {tflops:6.1f} TFLOPS")


if __name__ == "__main__":
    main()
