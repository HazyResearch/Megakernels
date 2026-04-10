from typing import List, Tuple

import torch
import torch.nn.functional as F

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorSpec
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


class Attention(IType):
    Mb = 128  # Q tile rows (seq dim)
    Db = 128  # head dim

    torch_functions = []
    torch_methods = ["attention"]

    # TMA tiles (axis=1 = DEPTH, tiling over seq_len × head_dim)
    Q_TMA = st(dtype=DType.bf16, rows=128, cols=128, axis=1)  # q_tile: st_bf<Mb, Db>
    K_TMA = st(dtype=DType.bf16, rows=64, cols=128, axis=1)   # k_tile: st_bf<Nb/2, Db>
    V_TMA = st(dtype=DType.bf16, rows=128, cols=64, axis=1)   # v_tile: st_bf<Nb, Db/2>
    O_TMA = st(dtype=DType.bf16, rows=128, cols=128, axis=1)  # o_tile: st_bf<Mb, Db>

    test_cases = [
        ((False,), (16, 1024, 16)), ((False,), (16, 2048, 16)), ((False,), (16, 4096, 16)),
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

    def block_indices(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> List[Tuple[int, ...]]:
        batch, seq_len, num_heads, _ = src_metas[0].shape
        num_block = seq_len // (self.Mb * self.TILES_PER_CLUSTER)

        if not self.causal:
            indices = []
            for b in range(batch):
                for h in range(num_heads):
                    for m in range(num_block):
                        indices.append((b, m, h))
                        indices.append((b, m, h))  # duplicate for CTA 1
            return indices

        else:
            # Causal: L2 swizzle + LPT ordering
            total_clusters = batch * num_heads * num_block
            size_one_head = seq_len * (self.Db + self.Db) * 2
            size_l2 = 50 * 1024 * 1024
            swizzle = 1
            if size_l2 >= size_one_head:
                swizzle = 1 << ((size_l2 // size_one_head).bit_length() - 1)

            num_hb = num_heads * batch
            num_hb_quotient = num_hb // swizzle
            num_hb_remainder = num_hb - num_hb_quotient * swizzle
            l2_major = swizzle * num_block

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
                b = bidhb_actual // num_heads
                h = bidhb_actual - b * num_heads
                m_cluster = num_block - 1 - m_cluster

                indices.append((b, m_cluster, h))
                indices.append((b, m_cluster, h))  # duplicate for CTA 1

            return indices

    def num_instructions(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> int:
        batch, seq_len, num_heads, _ = src_metas[0].shape
        m_blocks = seq_len // (self.Mb * self.TILES_PER_CLUSTER)
        return batch * num_heads * m_blocks * self.CLUSTER_SIZE

    def validate(self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...]) -> None:
        super().validate(src_metas, dst_metas)
        q, k, v = src_metas
        o = dst_metas[0]

        if len(q.shape) != 4:
            raise RuntimeError(f"[MegaKittens] Attention requires 4D tensors (BSHD), got {len(q.shape)}D")

        batch, seq_len, num_heads, head_dim = q.shape

        if head_dim != self.Db:
            raise RuntimeError(f"[MegaKittens] Attention requires head_dim={self.Db}, got {head_dim}")
        if k.shape != q.shape:
            raise RuntimeError(f"[MegaKittens] Attention requires Q and K same shape, got Q={q.shape} K={k.shape}")
        if v.shape != q.shape:
            raise RuntimeError(f"[MegaKittens] Attention requires Q and V same shape, got Q={q.shape} V={v.shape}")
        if o.shape != q.shape:
            raise RuntimeError(f"[MegaKittens] Attention output shape mismatch: expected {q.shape}, got {o.shape}")
