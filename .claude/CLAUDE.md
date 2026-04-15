# MegaKittens

GPU megakernel runtime that fuses PyTorch operator graphs into a single persistent CUDA kernel, built on top of [ThunderKittens](csrc/ThunderKittens/).

## Running tests

```
pytest
```

Tests disable JIT file cache (`use_jit_cache=False`) so they always recompile.

## Adding a new instruction type

Two files:

1. **`csrc/itypes/<name>.cuh`** — CUDA kernel struct with `controller`, `loader`, `launcher`, `consumer`, `storer` nested structs. Template params: `<Config, Globals, ...tensor indices>`. Access tensors via `g.template gls<I>()`. See `elementwise_unary.cuh` and `elementwise_binary.cuh` as reference.

2. **`megakittens/itypes/<name>.py`** — Subclass `IType`. Must define: custom op (`@torch.library.custom_op`), fake op, `inputs`/`outputs` (TensorSpec), `block_indices`, `num_instructions`, `validate`, `test_args`, `test_cases`. The scheduler auto-discovers via `IType.__subclasses__()`. The `__init_subclass__` auto-generates `test_fn` and registers the custom op in `torch_functions_map`.

### Python-side conventions

- Do not define small utility functions that are only used once. Inline the logic instead.
- Follow the code style of existing files as much as possible.
- `__init__` must support a default constructor (no required args).
- `test_cases` format: `[(cls_args_tuple, input_args_tuple), ...]`. `cls_args` is unpacked as `cls(*cls_args)`. The first test case should use the smallest valid shape to catch minimum-size edge cases.
- Custom ops that take variable ops use comma-separated strings (torch custom ops don't support `list[str]`).
- `torch_functions_map` values: `None` for plain IType registration, `callable` for resolvers that return an IType instance. The custom op itself needs a resolver if it has parameters (e.g. ops string).
- `cpp_template` must place `{tensors}` where tensor indices go. Non-int template params (enums, type lists) go before `{tensors}` since C++ can't have two parameter packs.
- `UNARY_OPS`/`BINARY_OPS` dicts map `str -> (cpp_enum_str, torch_callable)`.
- Granularity in TensorSpec only needs trailing dims (e.g. `(128, 128)` for a 2D tile). The base `validate` aligns from the right against the tensor shape.

### CUDA-side conventions

- All TMA coords are 4D: `{batch, depth, row, col}`. The gl pads lower-rank tensors to 4D automatically via `tensor_to_gl`.
- Instruction indices: `indices[0]` = batch, `indices[1]` = depth, then itype-specific (e.g. `indices[2]` = tile_row, `indices[3]` = tile_col, `indices[4]` = num_tiles).
- `block_indices` returns 4D+ tuples: `(b, d, r, c, ...)` matching the instruction indices layout.
- NVRTC JIT mode: functions must have `__device__` annotation. Use `__host__ __device__` for constexpr functions accessed at compile time. `static constexpr` arrays in structs are not allowed — use constexpr functions returning from local arrays instead.
- Elementwise ops use enum + constexpr if dispatch, not functors. Variadic ops use parameter packs with fold expressions.

## Architecture notes

- The icode dispatch switch is JIT-generated per graph (not in any header). Each unique `(itype, src_tensors, dst_tensors)` gets a fresh icode. The generated dispatch lives inside `namespace megakittens` along with `MKConfig` and `MKGlobals`.
- Tensor indices on instruction types are compile-time template params (e.g. `ElementwiseBinary<MKConfig, MKGlobals, BinaryOps<BinaryOp::ADD>, 0, 1, 2>`), not read from the instruction at runtime.
- `gls<I>()` on `MKGlobals` is a JIT-generated `if constexpr` chain returning the actual typed gl member. Each tensor keeps its original gl type (dtype, shape, TMA descriptors).
- `tensor_to_gl` left-pads tensor shapes to 4D: `(M, N)` → `(1, 1, M, N)`, `(B, M, N)` → `(1, B, M, N)`. The gl is always `gl<dtype, -1, -1, -1, -1>` for data tensors.
- Tensor ranges are absolute half-open slices of the backing tensor; `None` means full tensor. `block_indices` should schedule over the the effective range, while `access_regions` must return absolute backing-tensor regions for dependency barriers.
- `aot_autograd` flattens user inputs (including `list[Tensor]`) into individual tensors via pytree. String/scalar args (except for ints and floats, which become tensors) become graph constants. The dispatcher only sees flat tensors at runtime.
- Cross-SM barriers use global memory atomics. Source barriers check `target > 0` (unused targets are 0). Destination barriers use `0xFF` padding for unused slots.
- `compile_source_to_cubin` has both an in-memory `@functools.cache` and a file-backed cache (`~/.cache/megakittens/cubin/`). Pass `use_jit_cache=False` to skip file cache; in-memory cache still applies per-process.
- `NUM_PAGES` (currently 7 on Blackwell) is computed from shared memory in `csrc/schema.cuh` and mirrored as `Dispatcher.NUM_PAGES` in Python. Don't hardcode it.

## Rules for writing a CUDA instruction type (`csrc/itypes/*.cuh`)

- Every page claimed must be released. The loader must release unused pages ASAP and the consumer/storer must release the pages they used. The next instruction will stall until the pages it needs from the previous instruction are released.
- `lid_release_order` controls next-instruction page availability. Getting this right is essential to pipelining instructions.
- `init_semaphores` must return the count of semaphores initialized. The controller uses this to invalidate them between instructions. Getting this incorrect results in undefined behaviors.
- Use `page_t::as<T>()` to cast page data, not raw `reinterpret_cast`.
- Dependent template calls need `.template` keyword. E.g. `s.pages[pid].template as<tile_t>()`, `g.template gls<SRC0>()`. Forgetting this causes cryptic NVRTC "type name is not allowed" errors.
- `page_finish()` and `tensor_finish()` must be called by only one thread per instruction. The kernel will crash otherwise.
- Barrier wait/arrive are single-thread operations. Call `all_barrier_wait` and `all_barrier_arrive` from one elected thread only (e.g. inside `warp::elect_leader()`).
- Consumer is multiple warps unlike the other workers.
- TMA tile type in C++ must match the Python `TensorSpec.tma_types`. If the C++ `tile_t` is `st<bf16, 128, 128>` (swizzled), the Python side must create `st(dtype=DType.bf16, rows=128, cols=128)` (swizzle=True is default).
- Register tile height = shared tile height / group size. For `group<8>::load(rt, st)` with `st<128, 128>`, the rt must be `rt_bf<16, 128>` (128/8=16). Wrong dimensions cause a static assertion failure.

## Naming conventions

- "instruction type" / `IType` = how an op executes on the GPU (kernel implementation). Torch ops map to ITypes via `torch_functions_map`.
- "instruction" = one tile's worth of work dispatched to one SM
- "icode" = integer identifying which instruction type to dispatch (assigned per unique itype+tensor combo)
- Dimensions: `b` = batch, `d` = depth, `r` = row, `c` = col. Uppercase `B, D, R, C` for sizes, lowercase for indices.
