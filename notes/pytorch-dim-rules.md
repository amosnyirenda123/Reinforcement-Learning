# PyTorch Tensor Dimensions: The Complete Rulebook

## 0. The mental model you need first

Every tensor has a `shape`: a tuple like `(B, C, H, W)`. Every operation you call does exactly one of these four things to that shape:

1. **Preserves shape** — same shape in, same shape out (element-wise ops)
2. **Reduces/consumes a dimension** — a dimension shrinks or disappears (sum, mean, max...)
3. **Reshapes/restructures** — same total elements, different shape (view, permute, flatten...)
4. **Combines tensors** — two+ shapes interact via broadcasting or algebra (add, matmul, cat...)

Before calling *any* op on a tensor whose shape you're not 100% sure of, ask: **"which of these four buckets does this op belong to, and what does it do to `dim=X` specifically?"** That single habit eliminates most shape bugs.

Also internalize: **negative indexing on dims.** `dim=-1` is always the last dimension, `dim=-2` the second-to-last. This is used constantly and is not optional to know.

---

## 1. Operations that PRESERVE shape (element-wise)

Input shape == output shape, always. No dimension is touched.

- Arithmetic: `+ - * / torch.add torch.mul torch.pow`
- Activations: `relu, sigmoid, tanh, gelu, softplus, leaky_relu`
- Math: `exp, log, sqrt, abs, sin, cos, clamp`
- `torch.where(cond, a, b)` — preserves shape (after broadcasting cond/a/b to a common shape)

**The only trap here is broadcasting** (see §3) — the two operands don't need identical shapes, just *broadcast-compatible* ones. The output takes the broadcasted shape, not necessarily either input's original shape.

```python
a = torch.randn(4, 1, 8)
b = torch.randn(1, 5, 8)
(a + b).shape  # torch.Size([4, 5, 8])  <- NOT a's shape, NOT b's shape
```

---

## 2. Operations that CONSUME (reduce) a dimension

These are the #1 source of bugs because the output rank changes.

### The core reduction ops
`sum, mean, max, min, prod, std, var, median, all, any, argmax, argmin, norm, logsumexp`

**The rule that governs all of them:**

```python
x = torch.randn(32, 10, 256)   # (B, T, D)

x.sum()                 # scalar, shape ()          — all dims reduced
x.sum(dim=1)             # shape (32, 256)           — dim 1 GONE
x.sum(dim=1, keepdim=True)  # shape (32, 1, 256)     — dim 1 KEPT as size 1
x.sum(dim=(0, 1))        # shape (256,)              — multiple dims, both gone
```

- `dim` = which axis/axes get collapsed
- Without `keepdim=True`, that axis is **removed entirely** — everything after it shifts left
- With `keepdim=True`, the axis stays but becomes size 1 — **this is what you want 90% of the time in batch code**, because it keeps the rank stable for a subsequent broadcast (e.g. normalizing: `x / x.sum(dim=-1, keepdim=True)`)

**`max`/`min`/`argmax` with a `dim` argument are special**: they return a *tuple* `(values, indices)` (except `argmax`/`argmin` alone, which return just indices). Forgetting to unpack this is a classic bug:

```python
x.max(dim=1)              # returns namedtuple(values, indices), NOT a tensor
vals, idxs = x.max(dim=1)  # what you actually want
x.max(dim=1).values        # equivalent
```

### Matrix-reduction-like ops that also consume dims
- `torch.norm(x, dim=...)` — same keepdim behavior
- `torch.count_nonzero`, `torch.unique` (unique flattens by default unless `dim` given — collapses arbitrarily otherwise)

**Rule of thumb:** any function whose name describes an aggregate ("sum", "mean", "count", "max"...) is a reducer. Check its `dim` and `keepdim` args before using it in a batch pipeline.

---

## 3. Broadcasting — the exact rule (memorize this, not the intuition)

Two tensors are broadcastable if, aligning their shapes **from the right**:

> For each dimension pair, they are compatible if they are **equal**, or **one of them is 1**, or **one of them doesn't exist** (missing dims are treated as 1, implicitly prepended).

```
a: (   4, 1, 8)
b: (   1, 5, 8)
-------------------
out:(   4, 5, 8)     # each dim: max(compatible pair)

a: (3, 1)
b: (   4)
-------------------
out:(3, 4)            # b is treated as (1, 4), then broadcast
```

**Common failure**: `(32, 10)` + `(10,)` broadcasts fine (aligns from right, batch dim implicitly added). But `(32, 10)` + `(32,)` **fails** — because alignment is from the right, `(32,)` gets compared against the *last* dim (10), not the first (32). This is the #1 batch-processing broadcasting bug: forgetting that broadcasting aligns trailing dimensions, not leading ones.

**Fix**: use `unsqueeze` to explicitly place your dim where it needs to be:
```python
per_sample_scale = torch.randn(32)          # (32,) — meant to scale each row
x = torch.randn(32, 10)
# x * per_sample_scale        -> ERROR: shapes (32,10) and (32,) not broadcastable
x * per_sample_scale.unsqueeze(1)  # (32, 1) -> broadcasts correctly to (32, 10)
```

**Broadcasting never happens silently past this rule.** If it's not equal, not 1, and not missing — it's a hard error (`RuntimeError: The size of tensor a (...) must match the size of tensor b (...)`). That error message always tells you exactly which trailing dim mismatched — read it literally, it's not vague.

---

## 4. Operations applied "along one dimension" (not reductions — shape-preserving but dim-aware)

These don't change the shape, but their *behavior* is defined relative to one axis. Getting `dim` wrong here doesn't crash — it silently gives you wrong numbers, which is worse.

- `torch.softmax(x, dim=-1)` — normalizes so values along `dim` sum to 1. Shape unchanged.
- `torch.log_softmax`, `torch.nn.functional.normalize(x, dim=1)`
- `torch.cumsum(x, dim=...)`, `torch.cumprod`
- `torch.sort(x, dim=...)`, `torch.topk(x, k, dim=...)` — also return `(values, indices)` tuples like max
- `torch.nn.LayerNorm(normalized_shape)` — normalizes over the **last** `len(normalized_shape)` dims specifically (not a `dim` arg — you configure it at construction time)
- `BatchNorm` — normalizes over the batch dim (dim 0) **and** all dims except the channel dim (dim 1) — different convention, worth double-checking per-layer

**Rule of thumb**: for `softmax`, `normalize`, `cumsum` — always pass `dim` explicitly. Never rely on the default. In `(B, T, D)` sequence data, softmax over the vocabulary/feature axis is `dim=-1`; softmax over the time axis (e.g. attention weights) might be `dim=-1` too if attention is laid out as `(B, heads, T_query, T_key)` — softmax over keys is `dim=-1`. Always state out loud "softmax over the ___ axis" before writing the call.

---

## 5. Operations that RESHAPE (same elements, different arrangement)

These preserve total element count (`x.numel()` is invariant) but change shape/rank.

| Op | What it does | Gotcha |
|---|---|---|
| `view(*shape)` | Reinterprets shape, **no copy** | Requires the tensor to be contiguous in memory; fails after ops like `.transpose()`/`.permute()` without `.contiguous()` first |
| `reshape(*shape)` | Like `view` but copies if needed | Safer default than `view` when you're not sure about contiguity |
| `squeeze(dim)` | Removes size-1 dims (all of them if no `dim` given) | `squeeze()` with no arg can accidentally remove a batch dim if `B=1` — always pass explicit `dim` in production code |
| `unsqueeze(dim)` | Inserts a size-1 dim at position `dim` | Off-by-one on `dim` is extremely common — `unsqueeze(0)` adds batch dim, `unsqueeze(-1)` adds trailing dim |
| `permute(*dims)` | Reorders **all** dims arbitrarily | Must specify every dim's new position; different from transpose |
| `transpose(dim0, dim1)` | Swaps exactly two dims | Only two, unlike permute |
| `flatten(start_dim, end_dim)` | Collapses a range of dims into one | Default flattens everything; usually you want `flatten(start_dim=1)` to keep batch dim intact |
| `x.T` | Full reverse of dims (2D: normal transpose; 3D+: reverses all, rarely what you want) | Avoid on >2D tensors — use `permute` instead |

`-1` as a placeholder in `view`/`reshape` means "infer this dimension" — only one `-1` allowed per call:
```python
x = torch.randn(32, 10, 256)
x.view(32, -1)        # (32, 2560) — infers 2560 = 10*256
x.view(-1, 256)        # (320, 256) — infers 320 = 32*10, flattening batch+time together
```
That last pattern (`view(-1, D)`) is extremely common for feeding batched sequence data through a `Linear` layer that only knows about the last dim, then reshaping back.

---

## 6. Operations that COMBINE tensors (concat, stack, matmul)

### `cat` vs `stack` — the #1 confusion for beginners

```python
a = torch.randn(4, 10)
b = torch.randn(4, 10)

torch.cat([a, b], dim=0).shape    # (8, 10)   — dim 0 grows, rank stays 2
torch.stack([a, b], dim=0).shape  # (2, 4, 10) — NEW dim inserted, rank becomes 3
```
- `cat`: joins along an **existing** dimension. All other dims must match exactly.
- `stack`: creates a **new** dimension. All input tensors must have **identical** shape.

### Matrix multiplication family

- `torch.matmul` / `@` — the general-purpose one, follows these rules:
  - 2D @ 2D: standard matrix mult, `(n,m) @ (m,p) -> (n,p)`
  - Batched: `(B, n, m) @ (B, m, p) -> (B, n, p)` — leading dims are **batch dims** and are broadcast (not multiplied) if they differ, e.g. `(B,n,m) @ (m,p) -> (B,n,p)`
  - 1D involved: gets temporarily promoted/demoted (a 1D vector dotted into a matrix loses its "1" dim in the output) — this is the one truly confusing case, worth testing in isolation if unsure
- `torch.bmm` — strictly batched, requires exactly `(B,n,m) @ (B,m,p)`, no broadcasting, no implicit dim promotion. **Use this over matmul when you want the shape check to be strict** and catch bugs early.
- `torch.einsum` — you specify dims explicitly by letter, e.g. `torch.einsum('bnd,bmd->bnm', x, y)`. Slower to write but shape bugs become *impossible* to hide — the string itself is a shape contract. Worth using whenever you have >2 tensors or a dimension pattern that's hard to reason about with matmul/bmm alone.

**Batch matmul shape rule you'll use constantly** in attention-style code:
```python
q = torch.randn(B, heads, T, d_k)
k = torch.randn(B, heads, T, d_k)
scores = q @ k.transpose(-2, -1)   # (B, heads, T, d_k) @ (B, heads, d_k, T) -> (B, heads, T, T)
```
Note: only the *last two* dims participate in the actual matrix multiply; everything before that is treated as batch dims and must match (or broadcast).

---

## 7. Quick diagnostic checklist when you hit a shape error

1. **Read the actual error message fully** — PyTorch tells you the two conflicting shapes and (for broadcasting) which trailing dim failed. Don't skim past it.
2. **Print `.shape` right before the failing line**, not just at the top of the function. Shapes silently drift through reshape/permute/reduce chains.
3. Ask: is this op in bucket 1 (preserve), 2 (reduce), 3 (reshape), or 4 (combine)? That tells you what to check.
4. For broadcasting errors: align shapes from the **right**, check each pair is equal/1/missing.
5. For matmul errors: only the **last two dims** matter for the multiply itself; everything before must match as batch dims.
6. For `view` errors ("not contiguous"): you probably called `.transpose()`/`.permute()` upstream — add `.contiguous()` or switch to `.reshape()`.
7. For "wrong numbers, no crash" (the sneaky ones): almost always a `dim` argument to `softmax`/`normalize`/`sum` pointing at the wrong axis. Not a crash bug — a logic bug. Say out loud which axis you intend before writing `dim=`.

### A habit worth building
Add shape assertions inline during development — cheap insurance:
```python
assert x.shape == (B, T, D), f"expected (B,T,D)={B,T,D}, got {x.shape}"
```
Strip them out (or gate behind `if __debug__`) once the pipeline is stable.