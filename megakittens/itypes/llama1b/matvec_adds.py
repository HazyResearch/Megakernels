import torch

from ...schema.dtype import DType
from ...schema.itype import IType
from ...schema.tensor import TensorSpec
from ...jit.pykittens import sv, st


@torch.library.custom_op("megakittens::mat_vec_adds", mutates_args=())
def matvec_adds_op(
    x: torch.Tensor,
    down_weights: torch.Tensor,
) -> torch.Tensor:
    return down_weights[0] @ x


@matvec_adds_op.register_fake
def _matvec_adds_fake(x, down_weights):
    out_dim = down_weights.shape[1]
    return torch.empty(out_dim, dtype=x.dtype, device=x.device)


BLOCK_SIZE = 16


def _resolve_mat_vec_adds(args, kwargs):
    x = args[0].meta['val']
    return MatVecAdds(n=x.shape[-1])


class MatVecAdds(IType):

    torch_functions_map = {
        torch.ops.megakittens.mat_vec_adds: _resolve_mat_vec_adds,
        torch.ops.megakittens.mat_vec_adds.default: _resolve_mat_vec_adds,
    }

    test_cases = [
        ((2048,), (2048,)),  # (n,), (out_dim,)
    ]
    test_atol = 2.0
    test_rtol = 1e-2
    bench_cases = [
        ((2048,), (2048,)),
    ]

    def __init__(self, n=0):
        self._n = n

    @property
    def name(self) -> str:
        return "matvec_adds"

    @property
    def cpp_template(self) -> str:
        return f"MatVecAdds<MKConfig, MKGlobals, {self._n}, {{tensors}}>"

    @property
    def cpp_include(self) -> str:
        return "itypes/llama1b/matvec_adds.cuh"

    @property
    def op_type(self) -> str:
        return "matvec_adds"

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,)),                          # activations
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1),                      # weights
                       tma_types=[st(dtype=DType.bf16, rows=16, cols=512)]),
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,),                           # output (store_add)
                       tma_types=[sv(dtype=DType.bf16, length=16)]),
        ]

    def num_instructions(self, src_metas, dst_metas):
        out_dim = dst_metas[0].shape[0]
        num_blocks = out_dim // BLOCK_SIZE
        return 1 if num_blocks > 0 else 0

    def block_indices(self, src_metas, dst_metas):
        out_dim = dst_metas[0].shape[0]
        num_blocks = out_dim // BLOCK_SIZE
        if num_blocks == 0:
            return []
        return [(0, 0, num_blocks, 0)]

    def test_args(self, case):
        out_dim, = case
        n = self._n
        x = torch.randn(n, dtype=torch.bfloat16, device="cuda")
        down_weights = torch.randn(1, out_dim, n, dtype=torch.bfloat16, device="cuda")
        return (x, down_weights)

    def access_regions(self, block_index, src_metas, dst_metas):
        layer_idx, start_block, end_block, col_offset = block_index
        n = self._n
        x_region = ((col_offset, col_offset + n),)
        w_region = ((layer_idx, layer_idx + 1), (start_block * BLOCK_SIZE, end_block * BLOCK_SIZE), (col_offset, col_offset + n))
        out_region = ((start_block * BLOCK_SIZE, end_block * BLOCK_SIZE),)
        return [x_region, w_region], [out_region]

    def validate(self, src_metas, dst_metas):
        super().validate(src_metas, dst_metas)
        if src_metas[0].shape[-1] != self._n:
            raise RuntimeError(
                f"[MegaKittens] {self.name}: expected n={self._n}, got {src_metas[0].shape[-1]}"
            )
