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

    Args:
        fn: A plain PyTorch function (not yet compiled).
        args: Tuple of input tensors.
        atol: Absolute tolerance for allclose.
        rtol: Relative tolerance for allclose.

    Returns:
        (max_diff, mean_diff): Maximum and mean absolute difference.

    Raises:
        AssertionError if results don't match within tolerance.
    """
    torch._dynamo.reset()  # by default, dynamo limits to 8 compilations per function object
    compiled_fn = megakittens.compile(fn, use_jit_cache=False, save_dag=True, save_schedule=True)

    result = compiled_fn(*args)
    expected = fn(*args)

    diff = (result - expected).abs()
    max_diff: float = diff.max().item()
    mean_diff: float = diff.mean().item()

    if atol == 0.0 and rtol == 0.0:
        assert torch.equal(result, expected), \
            f"exact match failed: max_diff={max_diff}, mean_diff={mean_diff}"
    else:
        assert torch.allclose(result, expected, atol=atol, rtol=rtol), \
            f"allclose failed: max_diff={max_diff}, mean_diff={mean_diff}"

    return max_diff, mean_diff
