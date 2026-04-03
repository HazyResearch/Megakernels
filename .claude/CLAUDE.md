# MegaKittens

GPU megakernel runtime that fuses PyTorch operator graphs into a single persistent CUDA kernel, built on top of [ThunderKittens](csrc/ThunderKittens/).

## Environment

Always `conda activate enbao` first. Torch's bundled NVRTC 12.x was symlinked to NVRTC 13.0 (in `nvidia/cu13/lib/`) because B300 (`sm_103a`) requires NVRTC 13.0+. If a pip upgrade restores the old libs, re-symlink `nvidia/cuda_nvrtc/lib/libnvrtc.so.12` → `nvidia/cu13/lib/libnvrtc.so.13`.

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

## Rules for writing a CUDA instruction type (`csrc/itypes/*.cuh`)

- Every page claimed must be released. The loader must release unused pages ASAP and the consumer/storer must release the pages they used. The next instruction will stall until the pages it needs from the previous instruction are released.

- `lid_release_order` controls next-instruction page availability. Getting this right is essential to pipelining instructions.

- `init_semaphores` must return the count of semaphores initialized. The controller uses this to invalidate them between instructions. Getting this incorrect results in undefined behaviors.

- Use `page_t::as<T>()` to cast page data, not raw `reinterpret_cast`.

- Dependent template calls need `.template` keyword. E.g. `s.pages[pid].template as<tile_t>()`, `g.template gls<SRC0>()`. Forgetting this causes cryptic NVRTC "type name is not allowed" errors.

- `page_finish()` and `tensor_finish()` must be called by only one thread per instruction. The kernel will crash otherwise.

- Barrier wait/arrive are single-thread operations. Call `all_barrier_wait` and `all_barrier_arrive` from one elected thread only (e.g. inside `warp::elect_leader()`).

- Consumer is multiple warps unlike the other workers.

- **`fence.acquire` is per-warp, not per-SM.** If the launcher waits on a cross-SM barrier (`all_input_barrier_wait`), its `fence.acquire.gpu` only makes prior writes visible to the launcher warp. Other warps (e.g. the consumer) reading global memory written by a previous instruction must do their own barrier wait or fence. This was the cause of a stale-Q-read bug in `attention_partial` — the consumer loaded Q from `raw_ptr` without waiting for the QKV barrier, so it could read zeros/stale data.

- TMA tile type in C++ must match the Python `TensorSpec.tma_types`. If the C++ `tile_t` is `st<bf16, 128, 128>` (swizzled), the Python side must create `st(dtype=DType.bf16, rows=128, cols=128)` (swizzle=True is default).

- Register tile height = shared tile height / group size. For `group<8>::load(rt, st)` with `st<128, 128>`, the rt must be `rt_bf<16, 128>` (128/8=16). Wrong dimensions cause a static assertion failure.

## Naming conventions

- "op" / `OpType` = a vertex in the compute graph (DAG level, maps from torch ops)
- "instruction type" / `IType` = how an op executes on the GPU (kernel implementation)
- "instruction" = one tile's worth of work dispatched to one SM
- "icode" = integer identifying which instruction type to dispatch (assigned per unique itype+tensor combo)