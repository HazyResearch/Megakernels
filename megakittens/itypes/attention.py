from typing import List, Tuple

import torch
import torch.nn.functional as F

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorRange, TensorSpec
from ..jit.pykittens import st


@torch.library.custom_op("megakittens::attention", mutates_args=())
def attention_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    # q, k, v: (batch, seq_len, num_heads, head_dim) — BSHD layout
    q2 = q.transpose(1, 2)  # → (batch, num_heads, seq_len, head_dim)
    k2 = k.transpose(1, 2)
    v2 = v.transpose(1, 2)
    o = F.scaled_dot_product_attention(q2, k2, v2, is_causal=causal)
    return o.transpose(1, 2).contiguous()


@attention_op.register_fake
def _attention_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    return torch.empty_like(q)


def _resolve_attention(args, kwargs):
    causal = bool(args[3]) if len(args) > 3 else bool(kwargs.get("causal", False))
    return Attention(causal=causal)


def _resolve_aten_sdpa(args, kwargs):
    itype = Attention(causal=bool(args[6]) if len(args) > 6 else False)
    return itype, [0]


class Attention(IType):
    Mb = 128  # Q tile rows (seq dim)
    Db = 128  # head dim

    torch_functions_map = {
        torch.ops.aten._scaled_dot_product_cudnn_attention.default: _resolve_aten_sdpa,
        torch.ops.megakittens.attention: _resolve_attention,
        torch.ops.megakittens.attention.default: _resolve_attention,
    }
    torch_methods_map = {"attention": None}

    # TMA tiles (axis=1 = DEPTH, tiling over seq_len × head_dim)
    Q_TMA = st(dtype=DType.bf16, rows=128, cols=128, axis=1)  # q_tile: st_bf<Mb, Db>
    K_TMA = st(dtype=DType.bf16, rows=64, cols=128, axis=1)   # k_tile: st_bf<Nb/2, Db>
    V_TMA = st(dtype=DType.bf16, rows=128, cols=64, axis=1)   # v_tile: st_bf<Nb, Db/2>
    O_TMA = st(dtype=DType.bf16, rows=128, cols=128, axis=1)  # o_tile: st_bf<Mb, Db>

    test_cases = [
        ((False,), (1, 512, 1)),
        ((False,), (16, 1024, 16)), ((False,), (16, 2048, 16)), ((False,), (16, 4096, 16)),
        ((True,),  (1, 512, 1)),
        ((True,),  (16, 1024, 16)), ((True,),  (16, 2048, 16)), ((True,),  (16, 4096, 16)),
    ]
    test_atol = 1e-2
    test_rtol = 1e-2
    bench_cases = [
        ((False,), (16, 1024, 16)), ((False,), (16, 2048, 16)), ((False,), (16, 4096, 16)), ((False,), (16, 8192, 16)), ((False,), (16, 16384, 16)),
        ((True,),  (16, 1024, 16)), ((True,),  (16, 2048, 16)), ((True,),  (16, 4096, 16)), ((True,),  (16, 8192, 16)), ((True,),  (16, 16384, 16)),
    ]

    def __init__(self, causal: bool = False):
        self.causal = causal

    def test_args(self, case: tuple) -> tuple:
        B, S, H = case
        return (
            torch.randn(B, S, H, self.Db, dtype=torch.bfloat16, device="cuda"),
            torch.randn(B, S, H, self.Db, dtype=torch.bfloat16, device="cuda"),
            torch.randn(B, S, H, self.Db, dtype=torch.bfloat16, device="cuda"),
            self.causal,
        )

    def bench_flops(self, case: tuple) -> float:
        B, S, H = case
        return (2.0 if self.causal else 4.0) * B * H * S * S * self.Db

    @property
    def cpp_template(self) -> str:
        return f"Attention<MKConfig, MKGlobals, {{tensors}}, {'true' if self.causal else 'false'}>"

    @property
    def cpp_include(self) -> str:
        return "itypes/attention.cuh"

    @property
    def inputs(self) -> list[TensorSpec]:
        # Q, K, V: (batch, seq_len, num_heads, head_dim)
        seq_gran = self.Mb * self.TILES_PER_CLUSTER  # 512
        q_spec = TensorSpec(dtype=DType.bf16, granularity=(1, seq_gran, 1, self.Db), tma_types=[self.Q_TMA])
        k_spec = TensorSpec(dtype=DType.bf16, granularity=(1, seq_gran, 1, self.Db), tma_types=[self.K_TMA])
        v_spec = TensorSpec(dtype=DType.bf16, granularity=(1, seq_gran, 1, self.Db), tma_types=[self.V_TMA])
        return [q_spec, k_spec, v_spec]

    @property
    def outputs(self) -> list[TensorSpec]:
        # O: (batch, seq_len, num_heads, head_dim)
        seq_gran = self.Mb * self.TILES_PER_CLUSTER
        return [TensorSpec(dtype=DType.bf16, granularity=(1, seq_gran, 1, self.Db), tma_types=[self.O_TMA])]

    TILES_PER_CTA = 2       # consumer warpgroups per CTA
    CLUSTER_SIZE = 2
    TILES_PER_CLUSTER = TILES_PER_CTA * CLUSTER_SIZE  # 4

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        Q_range = src_ranges[0]
        K_range = src_ranges[1]
        V_range = src_ranges[2]
        O_range = dst_ranges[0]
        seq_block = self.Mb * self.TILES_PER_CLUSTER

        if not self.causal:
            indices = []
            for b in range(Q_range[0].size):
                for h in range(Q_range[2].size):
                    for m in range(Q_range[1].size // seq_block):
                        index = (
                            Q_range[0].start + b, Q_range[1].start // seq_block + m, Q_range[2].start + h,
                            K_range[0].start + b, K_range[2].start + h,
                            V_range[0].start + b, V_range[2].start + h,
                            O_range[0].start + b, O_range[1].start // seq_block + m, O_range[2].start + h,
                        )
                        indices.append(index)
                        indices.append(index)  # duplicate for CTA 1
            return indices

        else:
            # Causal: L2 swizzle + LPT ordering
            efficient_batch = Q_range[0].size
            efficient_heads = Q_range[2].size
            num_blocks = Q_range[1].size // seq_block
            total_clusters = efficient_batch * efficient_heads * num_blocks
            size_one_head = Q_range[1].size * (self.Db + self.Db) * 2
            size_l2 = 50 * 1024 * 1024
            swizzle = 1
            if size_l2 >= size_one_head:
                swizzle = 1 << ((size_l2 // size_one_head).bit_length() - 1)

            num_hb = efficient_heads * efficient_batch
            num_hb_quotient = num_hb // swizzle
            num_hb_remainder = num_hb - num_hb_quotient * swizzle
            l2_major = swizzle * num_blocks

            indices = []
            for cluster_linear in range(total_clusters):
                bidhb = cluster_linear // l2_major
                l2_mod = cluster_linear - bidhb * l2_major

                if bidhb < num_hb_quotient:
                    m_cluster = l2_mod // swizzle
                    bidhb_residual = l2_mod - m_cluster * swizzle
                else:
                    divisor = num_hb_remainder if num_hb_remainder > 0 else 1
                    m_cluster = l2_mod // divisor
                    bidhb_residual = l2_mod - m_cluster * divisor

                bidhb_actual = bidhb * swizzle + bidhb_residual
                b = bidhb_actual // efficient_heads
                h = bidhb_actual - b * efficient_heads
                m = num_blocks - 1 - m_cluster

                index = (
                    Q_range[0].start + b, Q_range[1].start // seq_block + m, Q_range[2].start + h,
                    K_range[0].start + b, K_range[2].start + h,
                    V_range[0].start + b, V_range[2].start + h,
                    O_range[0].start + b, O_range[1].start // seq_block + m, O_range[2].start + h,
                )
                indices.append(index)
                indices.append(index)  # duplicate for CTA 1

            return indices

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> int:
        Q_range = src_ranges[0]
        return Q_range[0].size * Q_range[2].size * (Q_range[1].size // (self.Mb * self.TILES_PER_CLUSTER)) * self.CLUSTER_SIZE

    def access_regions(self, block_index, src_metas, dst_metas):
        b_Q, m_Q, h_Q, b_K, h_K, b_V, h_V, b_O, m_O, h_O = block_index
        seq_block = self.Mb * self.TILES_PER_CLUSTER
        S_k = src_metas[1].shape[1]
        S_v = src_metas[2].shape[1]
        q_region = ((b_Q, b_Q + 1), (m_Q * seq_block, (m_Q + 1) * seq_block), (h_Q, h_Q + 1), (0, self.Db))
        k_region = ((b_K, b_K + 1), (0, S_k),                                 (h_K, h_K + 1), (0, self.Db))
        v_region = ((b_V, b_V + 1), (0, S_v),                                 (h_V, h_V + 1), (0, self.Db))
        o_region = ((b_O, b_O + 1), (m_O * seq_block, (m_O + 1) * seq_block), (h_O, h_O + 1), (0, self.Db))
        return [q_region, k_region, v_region], [o_region]

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        q, k, v = src_metas
        o = dst_metas[0]
        batch, seq_len, num_heads, head_dim = q.shape

        if head_dim != self.Db:
            raise RuntimeError(f"[MegaKittens] Attention requires head_dim={self.Db}, got {head_dim}")

        for label, meta, range in [("Q", q, src_ranges[0]), ("K", k, src_ranges[1]), ("V", v, src_ranges[2]), ("O", o, dst_ranges[0])]:
            if range[3].start != 0 or range[3].stop != meta.shape[3]:
                raise RuntimeError(
                    f"[MegaKittens] Attention requires full-head_dim range on {label}, got "
                    f"[{range[3].start}, {range[3].stop}) against head_dim={meta.shape[3]}"
                )
        for label, meta, range in [("K", k, src_ranges[1]), ("V", v, src_ranges[2])]:
            if range[1].start != 0 or range[1].stop != meta.shape[1]:
                raise RuntimeError(
                    f"[MegaKittens] Attention requires full-seq range on {label}, got "
                    f"[{range[1].start}, {range[1].stop}) against seq_len={meta.shape[1]}"
                )

        Q_effective_shape = src_ranges[0].effective_shape
        K_effective_shape = src_ranges[1].effective_shape
        V_effective_shape = src_ranges[2].effective_shape
        O_effective_shape = dst_ranges[0].effective_shape
        if Q_effective_shape != O_effective_shape:
            raise RuntimeError(f"[MegaKittens] Attention requires Q and O same effective shape, got Q={Q_effective_shape} O={O_effective_shape}")
        if K_effective_shape != V_effective_shape:
            raise RuntimeError(f"[MegaKittens] Attention requires K and V same effective shape, got K={K_effective_shape} V={V_effective_shape}")
        if Q_effective_shape[0] != K_effective_shape[0]:
            raise RuntimeError(f"[MegaKittens] Attention batch mismatch Q={Q_effective_shape[0]} K={K_effective_shape[0]}")
        if Q_effective_shape[2] != K_effective_shape[2]:
            raise RuntimeError(f"[MegaKittens] Attention num_heads mismatch Q={Q_effective_shape[2]} K={K_effective_shape[2]}")
