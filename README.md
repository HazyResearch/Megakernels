# MegaKittens

GPU megakernel runtime that fuses PyTorch operator graphs into a single persistent CUDA kernel, built on top of [ThunderKittens](https://github.com/HazyResearch/ThunderKittens).

## Requirements

MegaKittens requires pretty modern setup.

- NVIDIA Blackwell GPUs (B200/sm100a or B300/sm103a)
- CUDA 13+
- Python 3.12+
- PyTorch 2.9+

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

Using MegaKittens is as simple as putting `@megakittens.compile()` on top of your PyTorch functions! As long as the function follows our constraints (described in the next section), you should get out-of-the-box speedup immediately.

```python
import torch
import megakittens

@megakittens.compile()
def add(a, b):
    return a + b

a = torch.rand(128, 256, dtype=torch.bfloat16, device="cuda")
b = torch.rand(128, 256, dtype=torch.bfloat16, device="cuda")
result = add(a, b)
```

## Function Constraints

Work in progress.

## Tests

Each `tests/test_*.py` is runnable standalone (`python tests/test_add.py`). You can run all at once with `pytest`, although no actual dependency on `pytest` exists.

```bash
pytest                    # all tests
python -m tests.test_add  # single file
```

## Benchmarks

Each `benchmarks/benchmark_*.py` is a standalone script.

```bash
python -m benchmarks.benchmark_add
```

## Adding a new op

Only two files need to be touched: a new CUDA kernel in `csrc/ops/` and a Python descriptor in `megakittens/instruction.py`. See `add.cuh` and the `Add` class for reference. New ops are auto-discovered by the scheduler.
