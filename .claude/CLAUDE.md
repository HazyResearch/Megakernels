# MegaKittens

GPU megakernel runtime that fuses PyTorch operator graphs into a single persistent CUDA kernel, built on top of [ThunderKittens](csrc/ThunderKittens/).

## Running tests

```
pytest
```

Tests disable JIT file cache (`use_jit_cache=False`) so they always recompile.

## Adding a new instruction type

Only two files need to be touched:

1. **`csrc/itypes/<name>.cuh`** — CUDA kernel struct following the `MegaKittensIType` concept (must have `controller`, `loader`, `launcher`, `consumer`, `storer` nested structs). Template params: `<Config, Globals, ...tensor indices>`. Access tensors via `g.template gls<I>()`. See `add.cuh` as reference.

2. **`megakittens/itypes/<name>.py`** — Subclass `IType` with `name`, `op_type`, `cpp_template`, `cpp_include`, `inputs`, `outputs`, `block_indices`, `num_instructions`, `validate`. The scheduler auto-discovers it via `IType.__subclasses__()`.

## Architecture notes

- The icode dispatch switch is JIT-generated per graph (not in any header). Each unique `(itype, src_tensors, dst_tensors)` gets a fresh icode. The generated dispatch lives inside `namespace megakittens` along with `MKConfig` and `MKGlobals`.

- Tensor indices on instruction types are compile-time template params (e.g. `Add<Config, Globals, 3, 2, 4>`), not read from the instruction at runtime. This gives instruction types actual typed `gl` references for TMA/TK.

- `gls<I>()` on `MKGlobals` is a JIT-generated `if constexpr` chain returning the actual typed gl member. Each tensor keeps its original gl type (dtype, shape, TMA descriptors).

- Cross-SM barriers use global memory atomics. Source barriers check `target > 0` (unused targets are 0). Destination barriers use `0xFF` padding for unused slots.

- `compile_source_to_cubin` has both an in-memory `@functools.cache` and a file-backed cache (`~/.cache/megakittens/cubin/`). Pass `use_jit_cache=False` to skip file cache; in-memory cache still applies per-process.

## Naming conventions

- "op" / `OpType` = a vertex in the compute graph (DAG level, maps from torch ops)
- "instruction type" / `IType` = how an op executes on the GPU (kernel implementation)
- "instruction" = one tile's worth of work dispatched to one SM
- "icode" = integer identifying which instruction type to dispatch (assigned per unique itype+tensor combo)