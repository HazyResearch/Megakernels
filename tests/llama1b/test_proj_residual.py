"""
Test pipelined proj_residual instruction for both o_proj and down_proj.
Computes hidden_states += weights @ input and checks against PyTorch.

For o_proj (2048x2048): one instruction per output block, full reduction.
For down_proj (2048x8192): reduction split into 4 chunks of N=2048 each,
    with store_add_async accumulating partial results.
"""

import torch

from megakittens.jit.cuda_utils import initialize_cuda_context
from megakittens.dispatcher import Dispatcher, ScalarField
from megakittens.schema.device import Device
from megakittens.schema.dtype import DType
from megakittens.schema.instruction import Instruction, InstructionMeta
from megakittens.schema.tensor import TensorMeta
from megakittens.itypes.noop import Noop
from megakittens.itypes.proj_residual import ProjResidual

initialize_cuda_context()

HIDDEN_DIM = 2048
INTERMEDIATE_DIM = 8192
NUM_LAYERS = 16
BLOCK_SIZE = 16
N = 2048  # pipeline reduction capacity
DEVICE = "cuda"
DTYPE = torch.bfloat16
CLUSTER_SIZE = 2

T_INPUT = 0
T_WEIGHTS = 1
T_HIDDEN_STATES = 2

SCALARS = [
    ScalarField("pos_id", "unsigned int", 4, 4),
    ScalarField("attn_scale", "float", 4, 4),
    ScalarField("rms_norm_eps", "float", 4, 4),
]


def _run_proj_residual(input_vec, weights, hidden_states, layer_idx, output_dim,
                       reduction_dim, atol):
    """Run pipelined ProjResidual kernel and compare against PyTorch reference."""
    device = Device(type="cuda", index=0)

    expected = hidden_states.clone() + weights[layer_idx] @ input_vec

    tensor_metas = [
        TensorMeta(dtype=DType.bf16, shape=(reduction_dim,), device=device),
        TensorMeta(dtype=DType.bf16, shape=weights.shape, device=device),
        TensorMeta(dtype=DType.bf16, shape=(output_dim,), device=device),
    ]

    noop_itype = Noop()
    proj_itype = ProjResidual(n=N)

    instruction_metas = [
        InstructionMeta(icode=0, itype=noop_itype, src_tensors=(), dst_tensors=()),
        InstructionMeta(icode=1, itype=proj_itype,
                        src_tensors=(T_INPUT, T_WEIGHTS),
                        dst_tensors=(T_HIDDEN_STATES,)),
    ]

    noop = Instruction(
        icode=0, src_tensors=(), dst_tensors=(), indices=(),
        src_barriers=(), src_barrier_targets=(),
        num_input_barriers=0, num_reuse_barriers=0, num_dst_barriers=0,
        dst_barriers=(),
    )

    num_blocks = output_dim // BLOCK_SIZE
    num_chunks = reduction_dim // N  # 1 for o_proj, 4 for down_proj

    instructions = []
    for chunk in range(num_chunks):
        reduction_col_offset = chunk * N
        for block in range(num_blocks):
            instructions.append(Instruction(
                icode=1,
                src_tensors=(T_INPUT, T_WEIGHTS),
                dst_tensors=(T_HIDDEN_STATES,),
                indices=(layer_idx, block, block + 1, reduction_col_offset),
                src_barriers=(),
                src_barrier_targets=(),
                num_input_barriers=0,
                num_reuse_barriers=0,
                num_dst_barriers=0,
                dst_barriers=(),
            ))

    while len(instructions) % CLUSTER_SIZE != 0:
        instructions.append(noop)

    dispatcher = Dispatcher(
        instruction_metas, tensor_metas, instructions,
        num_barriers=0,
        input_tensor_indices=(T_INPUT, T_WEIGHTS, T_HIDDEN_STATES),
        output_tensor_indices=(T_HIDDEN_STATES,),
        use_jit_cache=False,
        scalar_fields=SCALARS,
    )

    result = dispatcher(input_vec, weights, hidden_states, 0, 0.125, 1e-5)
    torch.cuda.synchronize()

    assert result.data_ptr() == hidden_states.data_ptr(), "result is not in-place"

    diff = (result.float() - expected.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    return max_diff, mean_diff, atol


def test_o_proj_residual():
    """o_proj: hidden_states += o_weights[layer] @ attn_out  (2048x2048 @ 2048)"""
    torch.manual_seed(42)
    layer_idx = 3

    attn_out = torch.randn(HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    o_weights = torch.randn(NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM, dtype=DTYPE, device=DEVICE)
    hidden_states = torch.randn(HIDDEN_DIM, dtype=DTYPE, device=DEVICE)

    max_diff, mean_diff, atol = _run_proj_residual(
        attn_out, o_weights, hidden_states, layer_idx,
        output_dim=HIDDEN_DIM, reduction_dim=HIDDEN_DIM, atol=2.0,
    )
    print(f"o_proj: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    assert max_diff < atol, f"o_proj failed: max_diff={max_diff}"
    print("PASS")


def test_down_proj_residual():
    """down_proj: hidden_states += down_weights[layer] @ silu_out  (2048x8192 @ 8192)
    Reduction split into 4 chunks of 2048."""
    torch.manual_seed(42)
    layer_idx = 5

    silu_out = torch.randn(INTERMEDIATE_DIM, dtype=DTYPE, device=DEVICE)
    down_weights = torch.randn(NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM, dtype=DTYPE, device=DEVICE)
    hidden_states = torch.randn(HIDDEN_DIM, dtype=DTYPE, device=DEVICE)

    max_diff, mean_diff, atol = _run_proj_residual(
        silu_out, down_weights, hidden_states, layer_idx,
        output_dim=HIDDEN_DIM, reduction_dim=INTERMEDIATE_DIM, atol=4.0,
    )
    print(f"down_proj: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    assert max_diff < atol, f"down_proj failed: max_diff={max_diff}"
    print("PASS")


if __name__ == "__main__":
    test_o_proj_residual()
    test_down_proj_residual()
