import torch

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorSpec
from ...jit.pykittens import sv, st


@torch.library.custom_op("megakittens::rms_qkv_rope_append", mutates_args=())
def rms_qkv_rope_append_op(
    hidden_states: torch.Tensor, attn_norm_weights: torch.Tensor,
    qkv_weights: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    pos_id: torch.Tensor, rms_norm_eps: torch.Tensor,
) -> torch.Tensor:
    eps = rms_norm_eps.item()
    pos = pos_id.item()
    n = hidden_states.shape[-1]
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    num_attn_heads = n // head_dim
    q_dim = num_attn_heads * head_dim
    k_dim = num_kv_heads * head_dim
    h = torch.rms_norm(hidden_states, [n], attn_norm_weights[0], eps)
    qkv = qkv_weights[0] @ h
    q = qkv[:q_dim].view(num_attn_heads, head_dim)
    k = qkv[q_dim:q_dim + k_dim].view(num_kv_heads, head_dim)
    cos = rope_cos[pos]
    sin = rope_sin[pos]
    q_f = q.float()
    x1_q, x2_q = q_f[..., ::2], q_f[..., 1::2]
    q = (q_f * cos + torch.stack((-x2_q, x1_q), dim=-1).flatten(-2) * sin).to(q.dtype)
    k_f = k.float()
    x1_k, x2_k = k_f[..., ::2], k_f[..., 1::2]
    k = (k_f * cos + torch.stack((-x2_k, x1_k), dim=-1).flatten(-2) * sin).to(k.dtype)
    return q.reshape(-1)


@rms_qkv_rope_append_op.register_fake
def _rms_qkv_rope_append_fake(hidden_states, attn_norm_weights, qkv_weights,
                                rope_cos, rope_sin, k_cache, v_cache,
                                pos_id, rms_norm_eps):
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    q_dim = qkv_weights.shape[1] - 2 * num_kv_heads * head_dim
    return torch.empty(q_dim, dtype=hidden_states.dtype, device=hidden_states.device)


def _resolve_rms_qkv_rope_append(args, kwargs):
    hidden_states = args[0].meta['val']
    return RmsQkvRopeAppend(
        n=hidden_states.shape[-1],
        head_dim=args[3].meta['val'].shape[-1],       # rope_cos last dim
        num_kv_heads=args[5].meta['val'].shape[-2],   # k_cache second-to-last dim
    )


class RmsQkvRopeAppend(IType):

    torch_functions_map = {
        torch.ops.megakittens.rms_qkv_rope_append: _resolve_rms_qkv_rope_append,
        torch.ops.megakittens.rms_qkv_rope_append.default: _resolve_rms_qkv_rope_append,
    }

    test_cases = [
        ((2048, 64, 8), (0, 16)),  # (n, head_dim, num_kv_heads), (pos_id, max_seq_len)
    ]
    test_atol = 2.0
    test_rtol = 1e-2
    bench_cases = [
        ((2048, 64, 8), (0, 512)),
    ]

    def __init__(self, n=0, head_dim=64, num_kv_heads=8):
        self._n = n
        self._head_dim = head_dim
        self._num_kv_heads = num_kv_heads

    @property
    def name(self) -> str:
        return "rms_qkv_rope_append"

    @property
    def cpp_template(self) -> str:
        return (f"RmsQkvRopeAppend<MKConfig, MKGlobals, {self._n}, "
                f"{self._head_dim}, {self._num_kv_heads}, {{tensors}}>")

    @property
    def cpp_include(self) -> str:
        return "itypes/llama1b/rms_qkv_rope_append.cuh"

    @property
    def op_type(self) -> str:
        return "rms_qkv_rope_append"

    @property
    def inputs(self) -> list[TensorSpec]:
        if self._n > 0:
            return [
                TensorSpec(dtype=DType.bf16, granularity=(1,),                           # hidden_states
                           tma_types=[sv(dtype=DType.bf16, length=self._n)]),
                TensorSpec(dtype=DType.bf16, granularity=(1, 1),                         # attn_norm_weights
                           tma_types=[sv(dtype=DType.bf16, length=self._n)]),
                TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1),                      # qkv_weights
                           tma_types=[st(dtype=DType.bf16, rows=16, cols=512)]),
                TensorSpec(dtype=DType.fp32, granularity=(1, 1),                         # rope_cos
                           tma_types=[sv(dtype=DType.fp32, length=self._head_dim)]),
                TensorSpec(dtype=DType.fp32, granularity=(1, 1),                         # rope_sin
                           tma_types=[sv(dtype=DType.fp32, length=self._head_dim)]),
                TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1),                   # k_cache
                           tma_types=[sv(dtype=DType.bf16, length=16)]),
                TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1),                   # v_cache
                           tma_types=[sv(dtype=DType.bf16, length=16)]),
                TensorSpec(dtype=DType.int32, granularity=(1,)),                         # pos_id
                TensorSpec(dtype=DType.fp32, granularity=(1,)),                          # rms_norm_eps
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1)),
            TensorSpec(dtype=DType.fp32, granularity=(1, 1)),
            TensorSpec(dtype=DType.fp32, granularity=(1, 1)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1, 1)),
            TensorSpec(dtype=DType.int32, granularity=(1,)),
            TensorSpec(dtype=DType.fp32, granularity=(1,)),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,),                               # q_post_rope
                       tma_types=[sv(dtype=DType.bf16, length=16)]),
        ]

    def num_instructions(self, src_metas, dst_metas):
        qkv_dim = src_metas[2].shape[1]
        return 1 if qkv_dim >= 16 else 0

    def block_indices(self, src_metas, dst_metas):
        qkv_dim = src_metas[2].shape[1]
        num_blocks = qkv_dim // 16
        if num_blocks == 0:
            return []
        return [(0, 0, num_blocks)]

    def test_args(self, case):
        pos_id_val, max_seq_len = case
        n = self._n
        head_dim = self._head_dim
        num_kv_heads = self._num_kv_heads
        num_attn_heads = n // head_dim
        qkv_dim = (num_attn_heads + 2 * num_kv_heads) * head_dim
        hidden_states = torch.randn(n, dtype=torch.bfloat16, device="cuda")
        attn_norm_weights = torch.randn(1, n, dtype=torch.bfloat16, device="cuda")
        qkv_weights = torch.randn(1, qkv_dim, n, dtype=torch.bfloat16, device="cuda")
        rope_cos = torch.randn(max_seq_len, head_dim, dtype=torch.float32, device="cuda")
        rope_sin = torch.randn(max_seq_len, head_dim, dtype=torch.float32, device="cuda")
        k_cache = torch.zeros(1, max_seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        v_cache = torch.zeros(1, max_seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        pos_id = torch.tensor([pos_id_val], dtype=torch.int32, device="cuda")
        rms_norm_eps = torch.tensor([1e-5], dtype=torch.float32, device="cuda")
        return (hidden_states, attn_norm_weights, qkv_weights, rope_cos, rope_sin,
                k_cache, v_cache, pos_id, rms_norm_eps)

    def access_regions(self, block_index, src_metas, dst_metas):
        layer_idx, start_block, end_block = block_index
        n = src_metas[0].shape[0]
        max_seq_len, head_dim = src_metas[3].shape
        num_kv_heads = src_metas[5].shape[2]
        q_dim = (n // head_dim) * head_dim
        hidden_region = ((0, n),)
        norm_region = ((layer_idx, layer_idx + 1), (0, n))
        qkv_w_region = ((layer_idx, layer_idx + 1), (start_block * 16, end_block * 16), (0, n))
        rope_cos_region = ((0, max_seq_len), (0, head_dim))
        rope_sin_region = ((0, max_seq_len), (0, head_dim))
        k_cache_region = ((layer_idx, layer_idx + 1), (0, max_seq_len), (0, num_kv_heads), (0, head_dim))
        v_cache_region = ((layer_idx, layer_idx + 1), (0, max_seq_len), (0, num_kv_heads), (0, head_dim))
        pos_region = ((0, 1),)
        eps_region = ((0, 1),)
        out_region = ((0, q_dim),)
        return [hidden_region, norm_region, qkv_w_region, rope_cos_region, rope_sin_region,
                k_cache_region, v_cache_region, pos_region, eps_region], [out_region]

    def validate(self, src_metas, dst_metas):
        super().validate(src_metas, dst_metas)
        n = src_metas[0].shape[0]
        if n != self._n:
            raise RuntimeError(
                f"[MegaKittens] {self.name}: expected n={self._n}, got {n}"
            )
