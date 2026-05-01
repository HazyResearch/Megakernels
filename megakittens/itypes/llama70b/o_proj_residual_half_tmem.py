from typing import List, Tuple

import torch

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorMeta, TensorRange, TensorSpec
from ...jit.pykittens import st


Mb = 256
Kb = 64
DEFAULT_NB = 256
COLS_PER_CHUNK = 32
MIXED_MAIN_NB = 224
MIXED_TAIL_NB = 128


def _use_mixed_tail(m: int, n: int, nb: int) -> bool:
    return m == 512 and n == 8192 and nb == MIXED_MAIN_NB


def _resolve_nb(m: int, n: int, nb: int = 0) -> int:
    nb = int(nb)
    if nb == 0:
        nb = MIXED_MAIN_NB if m == 512 and n == 8192 else DEFAULT_NB
    if nb <= 0 or nb > 256 or nb % 32 != 0:
        raise RuntimeError(f"[MegaKittens] OProjResidualHalfTmem70b expected nb in 32..256 step 32, got {nb}")
    if nb % COLS_PER_CHUNK != 0:
        raise RuntimeError(f"[MegaKittens] OProjResidualHalfTmem70b requires nb divisible by {COLS_PER_CHUNK}, got {nb}")
    if n and n % nb != 0 and not _use_mixed_tail(m, n, nb):
        raise RuntimeError(f"[MegaKittens] OProjResidualHalfTmem70b requires N divisible by nb, got N={n}, nb={nb}")
    return nb


@torch.library.custom_op("megakittens::oproj_residual_half_tmem70b", mutates_args=("hidden",))
def o_proj_residual_half_tmem_op(
    hidden: torch.Tensor,
    attn_out: torch.Tensor,
    o_weights: torch.Tensor,
) -> None:
    hidden.add_(attn_out @ o_weights[0].transpose(-1, -2))


@o_proj_residual_half_tmem_op.register_fake
def _o_proj_residual_half_tmem_fake(
    hidden: torch.Tensor,
    attn_out: torch.Tensor,
    o_weights: torch.Tensor,
) -> None:
    pass


def _resolve_o_proj_residual_half_tmem(args, kwargs):
    hidden_node = args[0]
    attn_node = args[1]
    m = hidden_node.meta['val'].shape[-2]
    n = hidden_node.meta['val'].shape[-1]
    k = attn_node.meta['val'].shape[-1]
    return OProjResidualHalfTmem70b(m=m, n=n, k=k), [1]


class OProjResidualHalfTmem70b(IType):

    Mb = Mb
    Kb = Kb
    COLS_PER_CHUNK = COLS_PER_CHUNK

    A_TMA = st(dtype=DType.bf16, rows=Mb // 2, cols=Kb)

    torch_functions_map = {
        torch.ops.megakittens.oproj_residual_half_tmem70b: _resolve_o_proj_residual_half_tmem,
        torch.ops.megakittens.oproj_residual_half_tmem70b.default: _resolve_o_proj_residual_half_tmem,
    }

    test_cases = [
        ((256, 8192, 8192), (256, 8192, 8192)),
        ((512, 8192, 8192), (512, 8192, 8192)),
        ((1024, 8192, 8192), (1024, 8192, 8192)),
        ((2048, 8192, 8192), (2048, 8192, 8192)),
    ]
    test_atol = 1e-5
    test_rtol = 1e-5
    bench_cases = [
        ((512, 8192, 8192), (512, 8192, 8192)),
        ((1024, 8192, 8192), (1024, 8192, 8192)),
        ((2048, 8192, 8192), (2048, 8192, 8192)),
    ]

    @staticmethod
    def test_fn(hidden, attn_out, o_weights):
        torch.ops.megakittens.oproj_residual_half_tmem70b(hidden, attn_out, o_weights)
        return hidden

    def __init__(
        self,
        m: int = 0,
        n: int = 0,
        k: int = 0,
        nb: int = 0,
        mixed_tail=None,
        allow_partial_n: bool = False,
    ):
        self.m = m
        self.n = n
        self.k = k
        self.nb = _resolve_nb(m, n, nb)
        self.mixed_tail = _use_mixed_tail(m, n, self.nb) if mixed_tail is None else bool(mixed_tail)
        self.allow_partial_n = allow_partial_n

    @property
    def Nb(self) -> int:
        return self.nb

    @property
    def EPI_PIPE_DEPTH(self) -> int:
        return self.nb // self.COLS_PER_CHUNK

    @property
    def B_TMA(self):
        return st(dtype=DType.bf16, rows=self.Nb // 2, cols=self.Kb)

    @property
    def D_TMA(self):
        return st(dtype=DType.bf16, rows=self.Mb // 2, cols=self.COLS_PER_CHUNK)

    @property
    def cpp_template(self) -> str:
        return (
            f"llama70b::OProjResidualHalfTmem<MKConfig, MKGlobals, "
            f"{self.m}, {self.n}, {self.k}, "
            f"{self.Mb}, {self.Nb}, {self.Kb}, {self.EPI_PIPE_DEPTH}, "
            f"{{tensors}}>"
        )

    @property
    def cpp_include(self) -> str:
        return "itypes/llama70b/o_proj_residual_half_tmem.cuh"

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
            TensorSpec(dtype=DType.bf16, granularity=(self.Mb, self.COLS_PER_CHUNK),
                       tma_types=[self.D_TMA]),
            TensorSpec(dtype=DType.bf16, granularity=(self.Mb, self.Kb),
                       tma_types=[self.A_TMA]),
            TensorSpec(dtype=DType.bf16, granularity=(1, self.COLS_PER_CHUNK, self.Kb),
                       tma_types=[self.B_TMA]),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(self.Mb, self.COLS_PER_CHUNK),
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
        return len(self.block_indices(src_metas, dst_metas, src_ranges, dst_ranges))

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
        m_start = out_range[-2].start // self.Mb
        m_stop = out_range[-2].stop // self.Mb
        indices = []
        for m in range(m_start, m_stop):
            if self.mixed_tail:
                total_n = dst_metas[0].shape[-1]
                main_blocks = total_n // self.Nb
                tail_width = total_n - main_blocks * self.Nb
                for n in range(main_blocks):
                    col_start = n * self.Nb
                    if out_range[-1].start <= col_start and col_start + self.Nb <= out_range[-1].stop:
                        index = (layer_idx, m, n)
                        indices.append(index)
                        indices.append(index)
                if tail_width:
                    col_start = main_blocks * self.Nb
                    if out_range[-1].start <= col_start and col_start + tail_width <= out_range[-1].stop:
                        index = (layer_idx, m, col_start // tail_width)
                        indices.append(index)
                        indices.append(index)
            else:
                n_start = out_range[-1].start // self.Nb
                n_stop = out_range[-1].stop // self.Nb
                for n in range(n_start, n_stop):
                    index = (layer_idx, m, n)
                    indices.append(index)
                    indices.append(index)
        return indices

    def block_itype(
        self,
        block_index: Tuple[int, ...],
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ) -> IType:
        if not self.mixed_tail:
            return self
        _, _, n = block_index
        total_n = dst_metas[0].shape[-1]
        main_blocks = total_n // self.Nb
        tail_width = total_n - main_blocks * self.Nb
        nb = MIXED_TAIL_NB if tail_width and n == (main_blocks * self.Nb) // tail_width else self.Nb
        return type(self)(
            m=self.m,
            n=self.n,
            k=self.k,
            nb=nb,
            mixed_tail=False,
            allow_partial_n=True,
        )

    def access_regions(
        self,
        block_index: Tuple[int, ...],
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
    ):
        layer_idx, m, n = block_index
        K = src_metas[1].shape[-1]
        hidden_region = (
            (m * self.Mb, (m + 1) * self.Mb),
            (n * self.Nb, (n + 1) * self.Nb),
        )
        attn_out_region = (
            (m * self.Mb, (m + 1) * self.Mb),
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
                f"[MegaKittens] OProjResidualHalfTmem70b M mismatch: hidden M={M} vs attn_out M={a_shape[-2]}"
            )
        if w_shape[-2] != N:
            raise RuntimeError(
                f"[MegaKittens] OProjResidualHalfTmem70b N mismatch: hidden N={N} vs o_weights N={w_shape[-2]}"
            )
        if w_shape[-1] != K:
            raise RuntimeError(
                f"[MegaKittens] OProjResidualHalfTmem70b K mismatch: attn_out K={K} vs o_weights K={w_shape[-1]}"
            )

        if M % self.Mb != 0:
            raise RuntimeError(
                f"[MegaKittens] OProjResidualHalfTmem70b requires M divisible by {self.Mb}, got M={M}"
            )
        if N % self.COLS_PER_CHUNK != 0:
            raise RuntimeError(
                f"[MegaKittens] OProjResidualHalfTmem70b requires N divisible by {self.COLS_PER_CHUNK}, got N={N}"
            )
        if N % self.Nb != 0 and not (self.mixed_tail or self.allow_partial_n):
            raise RuntimeError(
                f"[MegaKittens] OProjResidualHalfTmem70b requires N divisible by {self.Nb}, got N={N}"
            )
        if K % self.Kb != 0:
            raise RuntimeError(
                f"[MegaKittens] OProjResidualHalfTmem70b requires K divisible by {self.Kb}, got K={K}"
            )

        if self.m and M != self.m:
            raise RuntimeError(
                f"[MegaKittens] OProjResidualHalfTmem70b expected M={self.m}, got {M}"
            )
        if self.n and N != self.n:
            raise RuntimeError(
                f"[MegaKittens] OProjResidualHalfTmem70b expected N={self.n}, got {N}"
            )
        if self.k and K != self.k:
            raise RuntimeError(
                f"[MegaKittens] OProjResidualHalfTmem70b expected K={self.k}, got {K}"
            )
