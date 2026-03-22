# MegaKittens

GPU megakernel runtime that fuses PyTorch operator graphs into a single persistent CUDA kernel, built on top of [ThunderKittens](csrc/ThunderKittens/).

## Running tests

```
pytest
```

Tests disable JIT file cache (`use_jit_cache=False`) so they always recompile.

## Adding a new op

Only two files need to be touched:

1. **`csrc/ops/<op>.cuh`** — CUDA kernel struct following the `MegaKittensOp` concept (must have `controller`, `loader`, `launcher`, `consumer`, `storer` nested structs). Template params: `<Config, Globals, ...tensor indices>`. Access tensors via `g.template gls<I>()`. See `add.cuh` as reference.

2. **`megakittens/instruction.py`** — Subclass `IType` with `name`, `tile_size`, `op_type`, `cpp_template`, `cpp_include`. The scheduler auto-discovers it via `IType.__subclasses__()`.

## Architecture notes

- The opcode dispatch switch is JIT-generated per graph (not in any header). Each unique `(itype, src_tensors, dst_tensors)` gets a fresh opcode. The generated dispatch lives inside `namespace megakittens` along with `MKConfig` and `MKGlobals`.

- Tensor indices on ops are compile-time template params (e.g. `Add<Config, Globals, 3, 2, 4>`), not read from the instruction at runtime. This gives ops actual typed `gl` references for TMA/TK operators.

- `gls<I>()` on `MKGlobals` is a JIT-generated `if constexpr` chain returning the actual typed gl member. Each tensor keeps its original gl type (dtype, shape, TMA descriptors).

- Cross-SM barriers use global memory atomics. Source barriers check `target > 0` (unused targets are 0). Destination barriers use `0xFF` padding for unused slots.

- `compile_source_to_cubin` has both an in-memory `@functools.cache` and a file-backed cache (`~/.cache/megakittens/cubin/`). Pass `use_jit_cache=False` to skip file cache; in-memory cache still applies per-process.