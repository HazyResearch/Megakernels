from typing import List, Tuple

import torch

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorMeta, TensorRange, TensorSpec
from ...jit.pykittens import sv, st


@torch.library.custom_op("megakittens::rms_upgate_silu", mutates_args=())
def rms_upgate_silu_op(
    x: torch.Tensor, norm_weight: torch.Tensor,
    up_weight: torch.Tensor, gate_weight: torch.Tensor,
    eps: torch.Tensor,
) -> torch.Tensor:
    e = eps.item()
    h = torch.rms_norm(x, [x.shape[-1]], norm_weight[0], e)
    return torch.nn.functional.silu(gate_weight[0] @ h) * (up_weight[0] @ h)


@rms_upgate_silu_op.register_fake
def _fake(x, norm_weight, up_weight, gate_weight, eps):
    out_dim = up_weight.shape[1]
    return torch.empty(out_dim, dtype=x.dtype, device=x.device)


def _resolve_rms_upgate_silu(args, kwargs):
    x = args[0].meta['val']
    return RmsUpgateSilu(n=x.shape[-1])


class RmsUpgateSilu(IType):

    torch_functions_map = {
        torch.ops.megakittens.rms_upgate_silu: _resolve_rms_upgate_silu,
        torch.ops.megakittens.rms_upgate_silu.default: _resolve_rms_upgate_silu,
    }

    test_cases = [
        ((2048,), (8192,)),  # (n,), (intermediate_dim,)
    ]
    test_atol = 1e-2
    test_rtol = 2e-2
    bench_cases = [
        ((2048,), (8192,)),
    ]

    def __init__(self, n=0):
        self._n = n

    @property
    def name(self) -> str:
        return "rms_upgate_silu"

    @property
    def cpp_template(self) -> str:
        return f"RmsUpgateSilu<MKConfig, MKGlobals, {self._n}, {{tensors}}>"

    @property
    def cpp_include(self) -> str:
        return "itypes/llama1b/upgate.cuh"

    @property
    def op_type(self) -> str:
        return "rms_upgate_silu"

    @property
    def inputs(self) -> list[TensorSpec]:
        if self._n > 0:
            return [
                TensorSpec(dtype=DType.bf16, granularity=(1,),                           # hidden_states
                           tma_types=[sv(dtype=DType.bf16, length=self._n)]),
                TensorSpec(dtype=DType.bf16, granularity=(1, 1),                         # norm_weight
                           tma_types=[sv(dtype=DType.bf16, length=self._n)]),
                TensorSpec(dtype=DType.bf16, granularity=(1, 16, 512),                    # up_weights
                           tma_types=[st(dtype=DType.bf16, rows=16, cols=512)]),
                TensorSpec(dtype=DType.bf16, granularity=(1, 16, 512),                    # gate_weights
                           tma_types=[st(dtype=DType.bf16, rows=16, cols=512)]),
                TensorSpec(dtype=DType.fp32, granularity=(1,)),                         # rms_norm_eps
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1)),
            TensorSpec(dtype=DType.fp32, granularity=(1,)),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(16,),                               # silu_out
                       tma_types=[sv(dtype=DType.bf16, length=16)]),
        ]

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> int:
        out_range = dst_ranges[0]
        return out_range[-1].size // 16

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        out_range = dst_ranges[0]
        layer_idx = src_ranges[2][-3].start
        num_blocks = out_range[-1].size // 16
        block_start = out_range[-1].start // 16
        block_stop = out_range[-1].stop // 16
        return [(layer_idx, b, num_blocks, num_blocks) for b in range(block_start, block_stop)]

    def test_args(self, case):
        intermediate_dim, = case
        n = self._n
        x = torch.randn(n, dtype=torch.bfloat16, device="cuda")
        norm_weight = torch.randn(1, n, dtype=torch.bfloat16, device="cuda")
        up_weight = torch.randn(1, intermediate_dim, n, dtype=torch.bfloat16, device="cuda")
        gate_weight = torch.randn(1, intermediate_dim, n, dtype=torch.bfloat16, device="cuda")
        eps = torch.tensor([1e-5], dtype=torch.float32, device="cuda")
        return (x, norm_weight, up_weight, gate_weight, eps)

    def access_regions(self, block_index, src_metas, dst_metas):
        layer_idx, sm_idx, sm_count, total_blocks = block_index
        n = src_metas[0].shape[0]
        intermediate_dim = dst_metas[0].shape[0]
        x_region = ((0, n),)
        norm_region = ((layer_idx, layer_idx + 1), (0, n))
        up_region = ((layer_idx, layer_idx + 1), (0, intermediate_dim), (0, n))
        gate_region = ((layer_idx, layer_idx + 1), (0, intermediate_dim), (0, n))
        eps_region = ((0, 1),)
        # one block per inst under auto: tight out region for fine-grained barriers
        out_region = ((sm_idx * 16, (sm_idx + 1) * 16),)
        return [x_region, norm_region, up_region, gate_region, eps_region], [out_region]

    def validate(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        if src_metas[0].shape[-1] != self._n:
            raise RuntimeError(
                f"[MegaKittens] {self.name}: expected n={self._n}, got {src_metas[0].shape[-1]}"
            )
