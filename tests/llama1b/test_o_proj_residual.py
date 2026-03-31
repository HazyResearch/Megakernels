"""
Test o_proj_residual instruction in isolation.
Computes hidden_states += o_weights @ attn_out and checks against PyTorch.
"""

import torch

from megakittens.jit.cuda_utils import get_sm_count, initialize_cuda_context
from megakittens.dispatcher import Dispatcher, ScalarField
from megakittens.schema.device import Device
from megakittens.schema.dtype import DType
from megakittens.schema.instruction import Instruction, InstructionMeta
from megakittens.schema.itype import IType
from megakittens.schema.tensor import TensorMeta, TensorSpec
from megakittens.itypes.noop import Noop

initialize_cuda_context()

HIDDEN_DIM = 2048
NUM_LAYERS = 16
BLOCK_SIZE = 16
DEVICE = "cuda"
DTYPE = torch.bfloat16


class OProjResidualIType(IType):
    @property
    def name(self): return "o_proj_residual"
    @property
    def cpp_template(self): return "OProjResidual<MKConfig, MKGlobals, {tensors}>"
    @property
    def cpp_include(self): return "itypes/o_proj_residual.cuh"
    @property
    def op_type(self): return "o_proj_residual"
    @property
    def inputs(self): return [
        TensorSpec(dtype=DType.bf16, granularity=(1,)),
        TensorSpec(dtype=DType.bf16, granularity=(1, 1, 1)),
    ]
    @property
    def outputs(self): return [TensorSpec(dtype=DType.bf16, granularity=(1,))]
    def block_indices(self, src_metas, dst_metas): return [()]
    def validate(self, src_metas, dst_metas): pass


# Tensor indices
T_ATTN_OUT = 0
T_O_WEIGHTS = 1
T_HIDDEN_STATES = 2

SCALARS = [
    ScalarField("pos_id", "unsigned int", 4, 4),
    ScalarField("attn_scale", "float", 4, 4),
    ScalarField("rms_norm_eps", "float", 4, 4),
]


def test_o_proj_residual():
    torch.manual_seed(42)
    device = Device(type="cuda", index=0)

    # Random test data for one layer
    attn_out = torch.randn(HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    o_weights = torch.randn(NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    hidden_states = torch.randn(HIDDEN_DIM, dtype=DTYPE, device=DEVICE)

    layer_idx = 3  # test an arbitrary layer

    # PyTorch reference
    expected = hidden_states.clone() + o_weights[layer_idx] @ attn_out

    # Build minimal schedule: 128 instructions, one per output block
    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=device),
        TensorMeta(dtype=DType.bf16, shape=(NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM), device=device),
        TensorMeta(dtype=DType.bf16, shape=(HIDDEN_DIM,), device=device),
    ]

    noop_itype = Noop()
    oproj_itype = OProjResidualIType()

    instruction_metas = [
        InstructionMeta(icode=0, itype=noop_itype, src_tensors=(), dst_tensors=()),
        InstructionMeta(icode=1, itype=oproj_itype,
                        src_tensors=(T_ATTN_OUT, T_O_WEIGHTS),
                        dst_tensors=(T_HIDDEN_STATES,)),
    ]

    num_blocks = HIDDEN_DIM // BLOCK_SIZE  # 128
    instructions = []
    for block in range(num_blocks):
        instructions.append(Instruction(
            icode=1,
            src_tensors=(T_ATTN_OUT, T_O_WEIGHTS),
            dst_tensors=(T_HIDDEN_STATES,),
            indices=(layer_idx, block, block + 1),
            src_barriers=(),
            src_barrier_targets=(),
            num_input_barriers=0,
            num_reuse_barriers=0,
            num_dst_barriers=0,
            dst_barriers=(),
        ))

    # Pad to cluster size
    noop = Instruction(
        icode=0, src_tensors=(), dst_tensors=(), indices=(),
        src_barriers=(), src_barrier_targets=(),
        num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
        dst_barriers=(),
    )
    while len(instructions) % 2 != 0:
        instructions.append(noop)

    input_indices = (T_ATTN_OUT, T_O_WEIGHTS, T_HIDDEN_STATES)
    output_indices = (T_HIDDEN_STATES,)

    dispatcher = Dispatcher(
        instruction_metas, tensor_metas, instructions,
        num_barriers=0,
        input_tensor_indices=input_indices,
        output_tensor_indices=output_indices,
        use_jit_cache=False,
        scalar_fields=SCALARS,
    )

    result = dispatcher(attn_out, o_weights, hidden_states, 0, 0.125, 1e-5)
    torch.cuda.synchronize()

    diff = (result.float() - expected.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    assert torch.allclose(result.float(), expected.float(), atol=1e-1, rtol=1e-1), \
        f"o_proj_residual failed: max_diff={max_diff}, mean_diff={mean_diff}"
    print("PASS")


if __name__ == "__main__":
    test_o_proj_residual()
