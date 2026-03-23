import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import megakittens


def benchmark(fn, args, warmup=10, iters=100):
    """
    Benchmark a function with and without MegaKittens compilation.

    Args:
        fn: A plain PyTorch function (not yet compiled).
        args: Tuple of input tensors.
        warmup: Number of warmup iterations.
        iters: Number of timed iterations.

    Returns:
        (mk_ms, pt_ms): Average time per call in milliseconds.
    """
    compiled_fn = megakittens.compile(fn, use_jit_cache=False, save_dag=True, save_schedule=True)

    # Warmup both
    for _ in range(warmup):
        compiled_fn(*args)
        fn(*args)
    torch.cuda.synchronize()

    # MegaKittens
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        compiled_fn(*args)
    end.record()
    torch.cuda.synchronize()
    mk_ms = start.elapsed_time(end) / iters

    # PyTorch baseline
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    pt_ms = start.elapsed_time(end) / iters

    return mk_ms, pt_ms
