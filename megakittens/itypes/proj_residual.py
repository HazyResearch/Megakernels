"""
Projection + residual add instruction type for decode.
Used for both o_proj and down_proj in Llama decode.

Computes: output[row] += dot(weights[layer][row], input)
"""

from typing import List, Tuple

import torch

from ..schema.dtype import DType
from ..schema.itype import IType
from ..schema.tensor import TensorMeta, TensorSpec


# Custom op registration (needed for optype.py import and megakittens.compile path)
@torch.library.custom_op("megakittens::proj_residual", mutates_args=())
def proj_residual_op(
    x: torch.Tensor,
    down_weights: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    return residual + x @ down_weights.T


@proj_residual_op.register_fake
def _proj_residual_fake(
    x: torch.Tensor,
    down_weights: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


BLOCK_SIZE = 16


class ProjResidual(IType):

    @property
    def name(self) -> str:
        return "proj_residual"

    @property
    def cpp_template(self) -> str:
        return "ProjResidual<MKConfig, MKGlobals, {tensors}>"

    @property
    def cpp_include(self) -> str:
        return "itypes/proj_residual.cuh"

    @property
    def op_type(self) -> str:
        return "proj_residual"

    @property
    def inputs(self) -> list[TensorSpec]:
        return [
            TensorSpec(dtype=DType.bf16, granularity=(1,)),      # input vector
            TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1)), # stacked weights
        ]

    @property
    def outputs(self) -> list[TensorSpec]:
        return [TensorSpec(dtype=DType.bf16, granularity=(1,))]  # output vector (in-place)

    def block_indices(
        self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...],
    ) -> List[Tuple[int, ...]]:
        return [()]

    def validate(
        self, src_metas: Tuple[TensorMeta, ...], dst_metas: Tuple[TensorMeta, ...],
    ) -> None:
        pass
