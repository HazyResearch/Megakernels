from typing import List, Tuple

import torch

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorMeta, TensorRange, TensorSpec
from ...jit.pykittens import sv, st


SM_COUNT = 148


@torch.library.custom_op("megakittens::rms_lm_head", mutates_args=())
def rms_lm_head_op(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    eps: torch.Tensor,
) -> torch.Tensor:
    e = eps.item()
    h = torch.rms_norm(x, [x.shape[-1]], norm_weight, e)
    return lm_head @ h


@rms_lm_head_op.register_fake
def _rms_lm_head_fake(x, norm_weight, lm_head, eps):
    vocab_size = lm_head.shape[0]
    return torch.empty(vocab_size, dtype=x.dtype, device=x.device)


def _resolve_rms_lm_head(args, kwargs):
    x = args[0].meta['val']
    return RmsLmHead(n=x.shape[-1])


class RmsLmHead(IType):

    torch_functions_map = {
        torch.ops.megakittens.rms_lm_head: _resolve_rms_lm_head,
        torch.ops.megakittens.rms_lm_head.default: _resolve_rms_lm_head,
    }

    test_cases = [
        ((2048,), (1024,)),  # (n,), (vocab_size,)
    ]
    test_atol = 1e-2
    test_rtol = 1e-2
    bench_cases = [
        ((2048,), (128256,)),
    ]

    def __init__(self, n=0):
        self._n = n

    @property
    def name(self) -> str:
        return "rms_lm_head"

    @property
    def cpp_template(self) -> str:
        return f"RmsLmHead<MKConfig, MKGlobals, {self._n}, {{tensors}}>"

    @property
    def cpp_include(self) -> str:
        return "itypes/llama1b/rms_lm_head.cuh"

    @property
    def op_type(self) -> str:
        return "rms_lm_head"

    @property
    def inputs(self) -> list[TensorSpec]:
        if self._n > 0:
            return [
                TensorSpec(dtype=DType.bf16, granularity=(1,),                           # hidden_states
                           tma_types=[sv(dtype=DType.bf16, length=self._n)]),
                TensorSpec(dtype=DType.bf16, granularity=(1,),                           # norm_weight
                           tma_types=[sv(dtype=DType.bf16, length=self._n)]),
                TensorSpec(dtype=DType.bf16, granularity=(16, 512),                       # lm_head_weights
                           tma_types=[st(dtype=DType.bf16, rows=16, cols=512)]),
                TensorSpec(dtype=DType.fp32, granularity=(1,)),                         # rms_norm_eps
            ]
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,)),
            TensorSpec(dtype=DType.bf16, granularity=(1,)),
            TensorSpec(dtype=DType.bf16, granularity=(1, 1)),
            TensorSpec(dtype=DType.fp32, granularity=(1,)),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(16,),                               # logits
                       tma_types=[sv(dtype=DType.bf16, length=16)]),
        ]

    def num_instructions(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> int:
        num_blocks = dst_ranges[0][-1].size // 16
        return min(SM_COUNT, num_blocks)

    def block_indices(
        self,
        src_metas: Tuple[TensorMeta, ...],
        dst_metas: Tuple[TensorMeta, ...],
        src_ranges: Tuple[TensorRange, ...],
        dst_ranges: Tuple[TensorRange, ...],
    ) -> List[Tuple[int, ...]]:
        out_range = dst_ranges[0]
        block_start = out_range[-1].start // 16
        block_stop = out_range[-1].stop // 16
        num_blocks = block_stop - block_start
        num_insts = min(SM_COUNT, num_blocks)
        return [
            (block_start + round(i * num_blocks / num_insts),
             block_start + round((i + 1) * num_blocks / num_insts))
            for i in range(num_insts)
        ]

    def test_args(self, case):
        vocab_size, = case
        n = self._n
        x = torch.randn(n, dtype=torch.bfloat16, device="cuda")
        norm_weight = torch.randn(n, dtype=torch.bfloat16, device="cuda")
        lm_head = torch.randn(vocab_size, n, dtype=torch.bfloat16, device="cuda")
        eps = torch.tensor([1e-5], dtype=torch.float32, device="cuda")
        return (x, norm_weight, lm_head, eps)

    def access_regions(self, block_index, src_metas, dst_metas):
        start_block, end_block = block_index
        n = src_metas[0].shape[0]
        x_region = ((0, n),)
        norm_region = ((0, n),)
        lm_head_region = ((start_block * 16, end_block * 16), (0, n))
        eps_region = ((0, 1),)
        out_region = ((start_block * 16, end_block * 16),)
        return [x_region, norm_region, lm_head_region, eps_region], [out_region]

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
