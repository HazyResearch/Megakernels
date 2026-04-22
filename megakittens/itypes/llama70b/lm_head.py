from typing import List, Tuple

import torch

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorMeta, TensorRange, TensorSpec
from ...jit.pykittens import st


Mb = 256
Nb = 256
Kb = 64
EPI_PIPE_DEPTH = 8
NUM_CONSUMERS = 2


@torch.library.custom_op("megakittens::lm_head70b", mutates_args=())
def lm_head70b_op(
    hidden: torch.Tensor,
    lm_head_weights: torch.Tensor,
) -> torch.Tensor:
    return hidden @ lm_head_weights[0].transpose(-1, -2)


@lm_head70b_op.register_fake
def _lm_head70b_fake(
    hidden: torch.Tensor,
    lm_head_weights: torch.Tensor,
) -> torch.Tensor:
    M = hidden.shape[-2]
    N = lm_head_weights.shape[-2]
    return torch.empty(M, N, dtype=hidden.dtype, device=hidden.device)


def _resolve_lm_head70b(args, kwargs):
    hidden_node = args[0]
    weights_node = args[1]
    m = hidden_node.meta['val'].shape[-2]
    k = hidden_node.meta['val'].shape[-1]
    n = weights_node.meta['val'].shape[-2]
    return LmHead70b(m=m, n=n, k=k)


class LmHead70b(IType):

    Mb = Mb
    Nb = Nb
    Kb = Kb
    EPI_PIPE_DEPTH = EPI_PIPE_DEPTH
    NUM_CONSUMERS = NUM_CONSUMERS
    M_INST = NUM_CONSUMERS * Mb

    A_TMA = st(dtype=DType.bf16, rows=Mb // 2, cols=Kb)
    B_TMA = st(dtype=DType.bf16, rows=Nb // 2, cols=Kb)
    D_TMA = st(dtype=DType.bf16, rows=Mb // 2, cols=Nb // EPI_PIPE_DEPTH)

    torch_functions_map = {
        torch.ops.megakittens.lm_head70b: _resolve_lm_head70b,
        torch.ops.megakittens.lm_head70b.default: _resolve_lm_head70b,
    }

    test_cases = [
        ((512, 128256, 8192), (512, 128256, 8192)),
        ((1024, 128256, 8192), (1024, 128256, 8192)),
        ((2048, 128256, 8192), (2048, 128256, 8192)),
    ]
    test_atol = 1e-5
    test_rtol = 1e-5
    bench_cases = [
        ((1024, 128256, 8192), (1024, 128256, 8192)),
    ]

    @staticmethod
    def test_fn(hidden, lm_head_weights):
        return torch.ops.megakittens.lm_head70b(hidden, lm_head_weights)

    def __init__(self, m: int = 0, n: int = 0, k: int = 0):
        self.m = m
        self.n = n
        self.k = k

    @property
    def cpp_template(self) -> str:
        return (
            f"llama70b::LmHead<MKConfig, MKGlobals, "
            f"{self.m}, {self.n}, {self.k}, "
            f"{self.Mb}, {self.Nb}, {self.Kb}, {self.EPI_PIPE_DEPTH}, "
            f"{{tensors}}>"
        )

    @property
    def cpp_include(self) -> str:
        return "itypes/llama70b/lm_head.cuh"

    def test_args(self, case: tuple) -> tuple:
        M, N, K = case
        hidden = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        lm_head_weights = torch.randn(1, N, K, dtype=torch.bfloat16, device="cuda") * (K ** -0.5)
        return (hidden, lm_head_weights)

    def bench_flops(self, case: tuple) -> float:
        M, N, K = case
        return 2.0 * M * N * K

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.M_INST, self.Kb),
                       tma_types=[self.A_TMA]),
            TensorSpec(dtype=DType.bf16, granularity=(1, self.Nb, self.Kb),
                       tma_types=[self.B_TMA]),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.M_INST, self.Nb),
                       tma_types=[self.D_TMA]),
        ]

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> int:
        out_range = dst_ranges[0]
        M = out_range[-2].size
        N = out_range[-1].size
        return 2 * (M // self.M_INST) * (N // self.Nb)

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        out_range = dst_ranges[0]
        m_start = out_range[-2].start // self.M_INST
        m_stop = out_range[-2].stop // self.M_INST
        n_start = out_range[-1].start // self.Nb
        n_stop = out_range[-1].stop // self.Nb
        indices = []
        for m in range(m_start, m_stop):
            for n in range(n_start, n_stop):
                index = (0, m, n)
                indices.append(index)
                indices.append(index)
        return indices

    def access_regions(
        self,
        block_index: Tuple[int, ...],
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ):
        _, m, n = block_index
        K = src_metas[0].shape[-1]
        hidden_region = (
            (m * self.M_INST, (m + 1) * self.M_INST),
            (0, K),
        )
        w_region = (
            (0, 1),
            (n * self.Nb, (n + 1) * self.Nb),
            (0, K),
        )
        logits_region = (
            (m * self.M_INST, (m + 1) * self.M_INST),
            (n * self.Nb, (n + 1) * self.Nb),
        )
        return (
            [[hidden_region], [w_region]],
            [[logits_region]],
        )

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        hidden_shape = src_ranges[0].effective_shape
        w_shape = src_ranges[1].effective_shape
        logits_shape = dst_ranges[0].effective_shape

        M = hidden_shape[-2]
        K = hidden_shape[-1]
        N = w_shape[-2]

        if w_shape[-1] != K:
            raise RuntimeError(
                f"[MegaKittens] LmHead70b K mismatch: hidden K={K} vs lm_head_weights K={w_shape[-1]}"
            )
        if logits_shape[-2] != M:
            raise RuntimeError(
                f"[MegaKittens] LmHead70b M mismatch: hidden M={M} vs logits M={logits_shape[-2]}"
            )
        if logits_shape[-1] != N:
            raise RuntimeError(
                f"[MegaKittens] LmHead70b N mismatch: lm_head_weights N={N} vs logits N={logits_shape[-1]}"
            )

        if M % self.M_INST != 0:
            raise RuntimeError(
                f"[MegaKittens] LmHead70b requires M divisible by {self.M_INST}, got M={M}"
            )
        if N % self.Nb != 0:
            raise RuntimeError(
                f"[MegaKittens] LmHead70b requires N divisible by {self.Nb}, got N={N}"
            )
        if K % self.Kb != 0:
            raise RuntimeError(
                f"[MegaKittens] LmHead70b requires K divisible by {self.Kb}, got K={K}"
            )

        if self.m and M != self.m:
            raise RuntimeError(
                f"[MegaKittens] LmHead70b expected M={self.m}, got {M}"
            )
        if self.n and N != self.n:
            raise RuntimeError(
                f"[MegaKittens] LmHead70b expected N={self.n}, got {N}"
            )
        if self.k and K != self.k:
            raise RuntimeError(
                f"[MegaKittens] LmHead70b expected K={self.k}, got {K}"
            )
