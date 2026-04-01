"""
Llama-1B decode plumbing test.

Verifies: scheduler -> dispatcher -> JIT compile -> kernel launch -> completion.
Uses noop instructions — no real compute, just tests the infrastructure.
"""

import torch

from megakittens.jit.cuda_utils import get_sm_count, initialize_cuda_context
from megakittens.dispatcher import Dispatcher, ScalarField
from megakittens.llama1b.scheduler import T, schedule_decode

initialize_cuda_context()

LLAMA1B_SCALARS = [
    ScalarField("pos_id", "unsigned int", 4, 4),
    ScalarField("attn_scale", "float", 4, 4),
    ScalarField("rms_norm_eps", "float", 4, 4),
]


def _allocate_tensors(tensor_metas):
    return [
        torch.zeros(meta.shape, dtype=meta.dtype.torch_dtype, device=str(meta.device))
        for meta in tensor_metas
    ]


def test_noop_single_layer():
    """One layer of noop instructions — minimal test of the full loop."""
    sm_count = get_sm_count()
    schedule = schedule_decode(sm_count=sm_count, num_layers=1, noop=True)
    instruction_metas, tensor_metas, instructions, num_barriers, input_indices, output_indices = schedule

    assert len(tensor_metas) == T.COUNT
    assert len(instructions) > 0
    assert num_barriers > 0
    print(f"Schedule: {len(instructions)} instructions, {num_barriers} barriers")

    dispatcher = Dispatcher(
        instruction_metas, tensor_metas, instructions, num_barriers,
        input_indices, output_indices,
        use_jit_cache=False,
        scalar_fields=LLAMA1B_SCALARS,
    )

    tensors = _allocate_tensors(tensor_metas)
    dispatcher(*tensors, 0, 0.125, 1e-5)
    torch.cuda.synchronize()
    print("Single layer noop: PASS")


def test_noop_full():
    """Full 16-layer noop schedule — tests barrier chaining across layers."""
    sm_count = get_sm_count()
    schedule = schedule_decode(sm_count=sm_count, num_layers=16, noop=True)
    instruction_metas, tensor_metas, instructions, num_barriers, input_indices, output_indices = schedule

    print(f"Schedule: {len(instructions)} instructions, {num_barriers} barriers")

    dispatcher = Dispatcher(
        instruction_metas, tensor_metas, instructions, num_barriers,
        input_indices, output_indices,
        use_jit_cache=False,
        scalar_fields=LLAMA1B_SCALARS,
    )

    tensors = _allocate_tensors(tensor_metas)
    dispatcher(*tensors, 0, 0.125, 1e-5)
    torch.cuda.synchronize()
    print("Full 16-layer noop: PASS")


if __name__ == "__main__":
    test_noop_single_layer()
    test_noop_full()
