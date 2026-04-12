from collections.abc import Callable

import torch
torch.manual_seed(42)

import megakittens


def check(
    fn: Callable[..., torch.Tensor],
    args: tuple[torch.Tensor, ...],
    atol: float = 0.0,
    rtol: float = 0.0,
) -> tuple[float, float]:
    """
    Run a function with and without MegaKittens compilation, compare results.
    Tests both per-SM queue and global work queue modes.

    Args:
        fn: A plain PyTorch function (not yet compiled).
        args: Tuple of input tensors.
        atol: Absolute tolerance for allclose.
        rtol: Relative tolerance for allclose.

    Returns:
        (max_diff, mean_diff): Maximum and mean absolute difference (from per-SM queue run).

    Raises:
        AssertionError if results don't match within tolerance.
    """
    torch._dynamo.reset()  # by default, dynamo limits to 8 compilations per function object
    expected = fn(*args)

    for global_work_queue in [True, False]:
        compiled_fn = megakittens.compile(fn, use_jit_cache=False, save_dag=False, save_schedule=False, verbose=False, global_work_queue=global_work_queue)
        result = compiled_fn(*args)

        diff = (result - expected).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        mode_str = "global_work_queue" if global_work_queue else "per_sm_queue"
        if atol == 0.0 and rtol == 0.0:
            assert torch.equal(result, expected), \
                f"[{mode_str}] exact match failed: max_diff={max_diff}, mean_diff={mean_diff}"
        else:
            assert torch.allclose(result, expected, atol=atol, rtol=rtol), \
                f"[{mode_str}] allclose failed: max_diff={max_diff}, mean_diff={mean_diff}"

    return max_diff, mean_diff
