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


@torch.library.custom_op("megakittens::oproj_residual70b", mutates_args=("hidden",))
def o_proj_residual_op(
    hidden: torch.Tensor,
    attn_out: torch.Tensor,
    o_weights: torch.Tensor,
) -> None:
    hidden.add_(attn_out @ o_weights[0].transpose(-1, -2))


@o_proj_residual_op.register_fake
def _o_proj_residual_fake(
    hidden: torch.Tensor,
    attn_out: torch.Tensor,
    o_weights: torch.Tensor,
) -> None:
    pass


def _resolve_o_proj_residual(args, kwargs):
    hidden_node = args[0]
    attn_node = args[1]
    m = hidden_node.meta['val'].shape[-2]
    n = hidden_node.meta['val'].shape[-1]
    k = attn_node.meta['val'].shape[-1]
    return OProjResidual70b(m=m, n=n, k=k), [1]


class OProjResidual70b(IType):

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
        torch.ops.megakittens.oproj_residual70b: _resolve_o_proj_residual,
        torch.ops.megakittens.oproj_residual70b.default: _resolve_o_proj_residual,
    }

    test_cases = [
        ((512, 8192, 8192), (512, 8192, 8192)),
        ((1024, 8192, 8192), (1024, 8192, 8192)),
        ((2048, 8192, 8192), (2048, 8192, 8192)),
        ((512, 8192, 28672), (512, 8192, 28672)),
        ((1024, 8192, 28672), (1024, 8192, 28672)),
    ]
    test_atol = 1e-5
    test_rtol = 1e-5
    bench_cases = [
        ((1024, 8192, 8192), (1024, 8192, 8192)),
    ]

    @staticmethod
    def test_fn(hidden, attn_out, o_weights):
        torch.ops.megakittens.oproj_residual70b(hidden, attn_out, o_weights)
        return hidden

    def __init__(self, m: int = 0, n: int = 0, k: int = 0):
        self.m = m
        self.n = n
        self.k = k

    @property
    def cpp_template(self) -> str:
        return (
            f"llama70b::OProjResidual<MKConfig, MKGlobals, "
            f"{self.m}, {self.n}, {self.k}, "
            f"{self.Mb}, {self.Nb}, {self.Kb}, {self.EPI_PIPE_DEPTH}, "
            f"{{tensors}}>"
        )

    @property
    def cpp_include(self) -> str:
        return "itypes/llama70b/o_proj_residual.cuh"

    def test_args(self, case: tuple) -> tuple:
        M, N, K = case
        hidden = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        attn_out = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        o_weights = torch.randn(1, N, K, dtype=torch.bfloat16, device="cuda") * (K ** -0.5)
        return (hidden, attn_out, o_weights)

    def bench_flops(self, case: tuple) -> float:
        M, N, K = case
        return 2.0 * M * N * K

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.M_INST, self.Nb),
                       tma_types=[self.D_TMA]),
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

    @property
    def inplace_mapping(self) -> dict[int, int]:
        return {0: 0}

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
        w_range = src_ranges[2]
        layer_idx = w_range[-3].start
        m_start = out_range[-2].start // self.M_INST
        m_stop = out_range[-2].stop // self.M_INST
        n_start = out_range[-1].start // self.Nb
        n_stop = out_range[-1].stop // self.Nb
        indices = []
        for m in range(m_start, m_stop):
            for n in range(n_start, n_stop):
                index = (layer_idx, m, n)
                indices.append(index)
                indices.append(index)
        return indices

    def access_regions(
        self,
        block_index: Tuple[int, ...],
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ):
        layer_idx, m, n = block_index
        K = src_metas[1].shape[-1]
        hidden_region = (
            (m * self.M_INST, (m + 1) * self.M_INST),
            (n * self.Nb, (n + 1) * self.Nb),
        )
        attn_out_region = (
            (m * self.M_INST, (m + 1) * self.M_INST),
            (0, K),
        )
        o_weights_region = (
            (layer_idx, layer_idx + 1),
            (n * self.Nb, (n + 1) * self.Nb),
            (0, K),
        )
        return (
            [[hidden_region], [attn_out_region], [o_weights_region]],
            [[hidden_region]],
        )

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        hidden_shape = dst_ranges[0].effective_shape
        a_shape = src_ranges[1].effective_shape
        w_shape = src_ranges[2].effective_shape

        M = hidden_shape[-2]
        N = hidden_shape[-1]
        K = a_shape[-1]

        if a_shape[-2] != M:
            raise RuntimeError(
                f"[MegaKittens] OProjResidual70b M mismatch: hidden M={M} vs attn_out M={a_shape[-2]}"
            )
        if w_shape[-2] != N:
            raise RuntimeError(
                f"[MegaKittens] OProjResidual70b N mismatch: hidden N={N} vs o_weights N={w_shape[-2]}"
            )
        if w_shape[-1] != K:
            raise RuntimeError(
                f"[MegaKittens] OProjResidual70b K mismatch: attn_out K={K} vs o_weights K={w_shape[-1]}"
            )

        if M % self.M_INST != 0:
            raise RuntimeError(
                f"[MegaKittens] OProjResidual70b requires M divisible by {self.M_INST}, got M={M}"
            )
        if N % self.Nb != 0:
            raise RuntimeError(
                f"[MegaKittens] OProjResidual70b requires N divisible by {self.Nb}, got N={N}"
            )
        if K % self.Kb != 0:
            raise RuntimeError(
                f"[MegaKittens] OProjResidual70b requires K divisible by {self.Kb}, got K={K}"
            )

        if self.m and M != self.m:
            raise RuntimeError(
                f"[MegaKittens] OProjResidual70b expected M={self.m}, got {M}"
            )
        if self.n and N != self.n:
            raise RuntimeError(
                f"[MegaKittens] OProjResidual70b expected N={self.n}, got {N}"
            )
        if self.k and K != self.k:
            raise RuntimeError(
                f"[MegaKittens] OProjResidual70b expected K={self.k}, got {K}"
            )
