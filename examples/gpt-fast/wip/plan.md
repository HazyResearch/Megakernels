# Plan: MegaKittens on gpt-fast LLaMA 3.2 1B

## Goal

Make `megakittens.compile` work end-to-end on the existing gpt-fast LLaMA 3.2 1B
inference code (`examples/gpt-fast/`). No new model code is written. All changes
happen in `megakittens/` (adding ITypes, extending `torch_functions_map`, etc.)
and in how/where `megakittens.compile` is applied within `generate.py` / `model.py`.

## Approach

Incrementally apply `megakittens.compile` to progressively larger sub-modules of
the existing model. At each step:

1. Wrap the target sub-module or sub-function with `megakittens.compile`.
2. Run inference and compare output to eager PyTorch baseline.
3. If compile fails on an unsupported ATen op — add support in `megakittens/`.
4. If output is numerically wrong — debug the IType implementation.
5. If correct — expand the compiled scope to include the next piece.
6. Repeat.

## Model structure (for reference)

```
Transformer.forward(mask, idx, input_pos):
    freqs_cis = self.freqs_cis[input_pos]
    x = self.tok_embeddings(idx)                          # Embedding
    for layer in self.layers:                              # 16 TransformerBlocks
        x = layer(x, input_pos, freqs_cis, mask)
    x = self.norm(x)                                      # RMSNorm
    logits = self.output(x)                                # Linear (no bias)
    return logits

TransformerBlock.forward(x, input_pos, freqs_cis, mask):
    h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
    out = h + self.feed_forward(self.ffn_norm(h))
    return out

Attention.forward(x, freqs_cis, mask, input_pos):
    q, k, v = self.wqkv(x).split(...)                     # Linear -> split
    q, k = apply_rotary_emb(q, freqs_cis), apply_rotary_emb(k, freqs_cis)
    q, k, v = [t.transpose(1,2) for t in (q, k, v)]       # BSHD -> BHSD
    k, v = self.kv_cache.update(input_pos, k, v)           # KV cache scatter
    y = flex_attention(q, k, v, block_mask=mask, enable_gqa=True)
    y = self.wo(y.transpose(1,2).contiguous().view(...))    # Linear
    return y

FeedForward.forward(x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))       # 3x Linear + SiLU + Mul

RMSNorm.forward(x):
    output = (x.float() * rsqrt(mean(x*x, dim=-1) + eps)).type_as(x)
    return output * self.weight
```

## Currently supported ITypes

| IType            | ATen ops handled                                         |
|------------------|----------------------------------------------------------|
| Gemm             | `aten.mm`, `aten.bmm`, `aten.matmul`                    |
| ElementwiseUnary | `aten.clone`, `relu`, `abs`, `exp`, `exp2`, `log`, `log2`, `neg`, `sqrt`, `rsqrt` |
| ElementwiseBinary| `aten.add.Tensor`, `sub`, `mul`, `div`, `maximum`, `minimum`, `atan2` |
| RMSNorm          | `aten._fused_rms_norm`                                   |
| Attention        | `aten._scaled_dot_product_cudnn_attention`               |

## Incremental steps

### Step 1 — Final Linear (`self.output`)

**Scope:** Compile only `self.output` (the vocab projection at the end of the
Transformer).

**Ops needed:** Gemm.

**Expected work:** None — Gemm is already supported. This validates the basic
compile + dispatch path on a single matmul with real model weights and shapes
(LLaMA 1B: `[B, 1, 2048] @ [2048, 128256]`).

---

### Step 2 — Final RMSNorm + Linear (`self.norm` → `self.output`)

**Scope:** Compile a function that applies `self.norm(x)` then `self.output(x)`.

**Ops needed:** RMSNorm → Gemm.

**Expected work:** Depends on how the hand-rolled RMSNorm traces. If it produces
`aten._fused_rms_norm`, no new work. If it decomposes into individual ops
(`_to_copy`, `mul`, `mean.dim`, `add.Scalar`, `rsqrt`, `type_as`), each must be
handled — likely requiring:
- Dtype cast IType or op (`aten._to_copy`)
- Scalar add support in ElementwiseBinary (or new op)
- Reduction mean (`aten.mean.dim`)

Alternative: swap the hand-rolled RMSNorm with `torch.nn.RMSNorm` or
`torch.rms_norm` so it emits `aten._fused_rms_norm` directly.

---

### Step 3 — FeedForward (`self.feed_forward`)

**Scope:** Compile `FeedForward.forward` in isolation.

**Ops needed:** Gemm (w1) + SiLU + Gemm (w3) + Mul + Gemm (w2).

**Expected work:**
- Add `silu` to `ElementwiseUnary.UNARY_OPS` (maps `aten.silu` → unary op).
- Corresponding CUDA enum entry in `csrc/itypes/elementwise_unary.cuh`.

---

### Step 4 — FFN half of TransformerBlock

**Scope:** Compile `ffn_norm(h)` → `feed_forward(...)` → residual add.

**Ops needed:** RMSNorm + Gemm + SiLU + Mul + Gemm + Add.

**Expected work:** Composition of steps 2 and 3. Validates RMSNorm → MLP → Add
chaining with real weights.

---

### Step 5 — Simplified Attention (no rotary, no KV cache, no flex_attention)

**Scope:** Compile a stripped-down attention path: QKV projection → standard SDPA
→ output projection. Temporarily bypass rotary embeddings, KV cache, and
flex_attention to validate the core attention data flow.

**Ops needed:** Gemm (wqkv) + split/view/transpose + Attention (SDPA) +
transpose/contiguous/view + Gemm (wo).

**Expected work:**
- Verify view/transpose/split ops are handled by the tracer (they should be — the
  tracer already resolves `getitem`, `select`, `slice`).
- May need to handle `aten.split.Tensor` or `aten.split_with_sizes`.
- Validate the existing Attention IType works with GQA (32 Q heads, 8 KV heads)
  and BHSD layout. The current IType expects BSHD — may need layout adaptation.

---

### Step 6 — Attention + rotary embeddings

**Scope:** Add `apply_rotary_emb` back into the attention path.

**Ops needed:** Reshape + elementwise mul/sub/add + stack + flatten (the RoPE
computation).

**Expected work:** The RoPE function does:
```python
xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
x_out = stack([
    xshaped[...,0]*freqs[...,0] - xshaped[...,1]*freqs[...,1],
    xshaped[...,1]*freqs[...,0] + xshaped[...,0]*freqs[...,1],
], -1).flatten(3).type_as(x)
```
This decomposes into: dtype cast, reshape (view), indexing (select on last dim),
mul, sub, add, stack, flatten, dtype cast. Several of these may need new support:
- `aten.stack` — new IType or view handling
- `aten.select.int` on inner dims — tracer handles this for edges but the op
  itself needs an IType if it appears as a compute node
- Dtype casts (`aten._to_copy`)

Alternatively, consider a fused RotaryEmbedding IType.

---

### Step 7 — Attention + KV cache

**Scope:** Add `KVCache.update` back into the attention path.

**Ops needed:** Index scatter (`k_cache[:, :, input_pos] = k_val`).

**Expected work:** New IType for `aten.index_put` or `aten.scatter` — an indexed
write operation.

---

### Step 8 — Attention with flex_attention

**Scope:** Replace standard SDPA with `flex_attention` + `BlockMask` + GQA.

**Ops needed:** `flex_attention` kernel dispatch.

**Expected work:** This is the hardest step. `flex_attention` is a higher-order op
with a mask_mod callback. Options:
- New Attention IType variant that handles flex_attention semantics.
- Decompose flex_attention into standard attention + masking that MegaKittens can
  handle.
- Hybrid approach: let flex_attention run via its own Triton kernel and compile
  the surrounding ops with MegaKittens.

---

### Step 9 — Full TransformerBlock

**Scope:** Compile an entire `TransformerBlock.forward`.

**Ops needed:** All of the above combined: RMSNorm → Attention (with rotary, KV
cache, flex_attention) → Add → RMSNorm → FeedForward → Add.

**Expected work:** Primarily a composition/integration test. May surface
barrier/scheduling issues when all ITypes are chained in a single megakernel.

---

### Step 10 — Full Transformer

**Scope:** Compile the entire `Transformer.forward` end-to-end.

**Ops needed:** Embedding + 16 × TransformerBlock + RMSNorm + Linear.

**Expected work:**
- New IType for `aten.embedding` (table lookup).
- Handle the `freqs_cis[input_pos]` indexing (gather by position).
- Scale test: 16 layers means many instructions and tensors — may hit scheduler
  limits or performance issues.

---

## Validation at each step

At every step, correctness is verified by:
1. Running the compiled sub-module on the same input as the eager baseline.
2. Comparing outputs with `torch.allclose` (tolerances appropriate for bf16).
3. Running full text generation and confirming coherent output matches the eager
   model.

## Files modified

- `megakittens/itypes/` — new ITypes and extended op mappings.
- `csrc/itypes/` — corresponding CUDA kernel structs.
- `examples/gpt-fast/generate.py` — where `megakittens.compile` is applied
  (scope changes per step).
- `examples/gpt-fast/model.py` — possibly minor changes to control which
  sub-modules are compiled (but no model logic changes).
