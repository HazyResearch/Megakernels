"""Smoke + microbench for the gate_silu_forward binding.

Reference path mirrors megakittens/itypes/llama70b/gate_silu.py:
out = silu(x @ gate_w[0].T).
"""

from __future__ import annotations

try:
    import pytest
except ImportError:
    pytest = None

import torch

from . import _C


HIDDEN_DIM = 8192
INTERMEDIATE_DIM = 28672


def _ref(x: torch.Tensor, gate_w: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(
        (x.float() @ gate_w[0].float().transpose(-1, -2))
    ).to(x.dtype)


def _make_case(M: int, N: int, K: int, device: str = "cuda"):
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    gate_w = torch.randn(1, N, K, dtype=torch.bfloat16, device=device) * (K ** -0.5)
    return x, gate_w


_CASES = [
    (512,  256,  HIDDEN_DIM),     # smallest valid: M=M_INST, N=Nb
    (512,  INTERMEDIATE_DIM, HIDDEN_DIM),
    (1024, INTERMEDIATE_DIM, HIDDEN_DIM),
    (2048, INTERMEDIATE_DIM, HIDDEN_DIM),
]


if pytest is not None:
    @pytest.mark.parametrize("M,N,K", _CASES)
    def test_gate_silu(M, N, K):
        torch.manual_seed(0)
        x, gate_w = _make_case(M, N, K)
        ref = _ref(x, gate_w)
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        _C.gate_silu_forward(x, gate_w, out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def main():
    torch.manual_seed(0)
    for M, N, K in _CASES:
        x, gate_w = _make_case(M, N, K)
        ref = _ref(x, gate_w)
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        _C.gate_silu_forward(x, gate_w, out)
        torch.cuda.synchronize()
        err = (out.float() - ref.float()).abs()

        warmup, iters = 20, 200
        for _ in range(warmup):
            _C.gate_silu_forward(x, gate_w, out)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _C.gate_silu_forward(x, gate_w, out)
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000.0 / iters
        tflops = 2.0 * M * N * K / (us * 1e-6) / 1e12
        print(f"M={M:5d} N={N:6d} K={K:5d}  err={err.max().item():.4g}  "
              f"{us:7.2f} us  {tflops:6.1f} TFLOPS")


if __name__ == "__main__":
    main()
