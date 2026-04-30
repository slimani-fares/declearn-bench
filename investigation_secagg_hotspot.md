# SecAgg masking hotspot — `_generate_masks_numpy` investigation

Read-only investigation. No source files were modified.
Date: 2026-04-30.

All paths below are relative to `~/work/declearn-bench/declearn-for-secagg/`.

---

## 1. Full source — `_generate_masks_numpy`

**File:** `declearn/secagg/masking/_encrypt.py`
**Lines:** 123–134

```python
123    def _generate_masks_numpy(
124        self,
125        n_values: int,
126    ) -> np.ndarray:
127        """Generate a number of masking values."""
128        mask = np.zeros(shape=(n_values,), dtype=self._dtype)
129        max_val = self.max_int
130        for rng in self._pos_rng:
131            mask += rng.integers(max_val, dtype=self._dtype, size=n_values)
132        for rng in self._neg_rng:
133            mask -= rng.integers(max_val, dtype=self._dtype, size=n_values)
134        return mask
```

The user-cited "lines 131 and 133" map exactly to the two `rng.integers(...)` ufunc calls (one for the positive-mask peers, one for the negative-mask peers). The `max_val = self.max_int` on line 129 is just a local-name binding to avoid attribute lookup inside the loops.

## 2. Immediate caller — `encrypt_uint`

**File:** `declearn/secagg/masking/_encrypt.py`
**Lines:** 136–141

```python
136    def encrypt_uint(
137        self,
138        value: int,
139    ) -> int:
140        mask = int(self._generate_masks(1)[0])
141        return (value + mask) % self.max_int
```

Two things to note:

- It calls `self._generate_masks(1)` — i.e. it asks for **a single mask value**, then indexes `[0]` and converts to a Python `int`. So every scalar that goes through encryption triggers one full pass through `_generate_masks_numpy` with `n_values=1`.
- `self._generate_masks` is bound at `__init__` time to either `_generate_masks_numpy` (line 102) or `_generate_masks_large` (line 100), depending on whether the configured `bitsize` fits in a numpy uint dtype. For your `bitsize=64` config it binds to the numpy path.

## 3. Next caller up — `encrypt_vector`

**File:** `declearn/secagg/api/_encrypt.py`
**Lines:** 157–178

```python
157    def encrypt_vector(
158        self,
159        value: Vector,
160    ) -> Tuple[List[int], VectorSpec]:
...
175        flt_val, v_spec = value.flatten()
176        int_val = self.quantizer.quantize_list(flt_val)
177        enc_val = [self.encrypt_uint(val) for val in int_val]
178        return enc_val, v_spec
```

Line 177 is the smoking gun: a Python-level list comprehension that calls `encrypt_uint` **once per scalar element of the flattened vector**. Each of those calls fires off `_generate_masks_numpy(1)`.

For a torch model with `L` parameters, that's `L` invocations of `_generate_masks_numpy(1)` per client per round.

## 4. Specific answers

### (a) The two dominant statements at lines 131 and 133

```python
131    mask += rng.integers(max_val, dtype=self._dtype, size=n_values)
133    mask -= rng.integers(max_val, dtype=self._dtype, size=n_values)
```

Both share the same shape: a numpy `Generator.integers` PRNG draw of size `n_values` (which is **always 1** when called from `encrypt_uint`), bounded by `self.max_int = 2**bitsize`, dtype `self._dtype` (`uint64` for `bitsize=64`), then fused-added or subtracted into the running `mask` array via numpy ufunc broadcasting.

The work that py-spy is sampling on these lines is:
- `rng.integers(...)` — invokes numpy's PCG64 (default since numpy ≥ 1.17) C-side stream + a per-call argument-validation/dtype-dispatch path on the Python side.
- The `+=`/`-=` is a numpy in-place ufunc; on a 1-element array it is essentially pure call overhead.

For `n_values=1`, the per-call fixed Python/numpy overhead **dominates** the actual byte-of-randomness produced. This is the classic vectorized-API-called-in-a-scalar-loop antipattern.

### (b) Where does `numpy.prod()` come from?

**There is no direct `np.prod` / `numpy.prod` call inside `_generate_masks_numpy`.** I greped `masking/_encrypt.py` and confirmed zero occurrences:

```bash
$ grep -n "np\.prod\|numpy\.prod" declearn/secagg/masking/_encrypt.py
(no output)
```

If py-spy is attributing self time to `numpy.prod` while the call stack is rooted in `_generate_masks_numpy`, it is coming from numpy **internals**, not user code. Two specific spots inside numpy that compute a shape-product on every call:

1. **`np.zeros(shape=(n_values,), dtype=self._dtype)` on line 128.** Numpy's array allocator computes the total element count by multiplying the shape tuple's entries — for a length-1 shape tuple this still goes through the `shape -> nbytes` machinery that ends up calling `numpy.core.multiarray._array_from_buffer` or similar, which internally evaluates a product over the shape. py-spy's pure-Python sampler can attribute frames here as `numpy.prod` because numpy's pure-Python helpers (`numpy.core.fromnumeric.prod`) are visible to the sampler when they are on the Python stack at sample time.

2. **`rng.integers(..., size=n_values)` on lines 131 and 133.** `Generator.integers` resolves the `size` argument through numpy's broadcasting helpers, which compute total output size as a product of the size tuple. This is the most likely top-of-stack target for the `np.prod` attribution because it fires twice per loop body, on every invocation, and the loop runs `n_peers - 1` times per `_generate_masks_numpy(1)` call.

Either way, **`np.prod` is being called many times per invocation** — once per `np.zeros` plus once per `rng.integers` (so at least `1 + (len(pos_rng) + len(neg_rng))` shape-products per call, with `n_values=1` the whole time). It is not a one-shot cost.

### (c) Does the function receive shape, or derive it?

`_generate_masks_numpy` takes `n_values: int` as a parameter, **not** an input array. It then constructs its own zero-filled mask array on line 128:

```python
128    mask = np.zeros(shape=(n_values,), dtype=self._dtype)
```

It is always called with the literal `1` from `encrypt_uint` (line 140), so the shape `(1,)` is **constant across every call within a round** — and across every call across the entire run, in fact. Every single mask allocation re-creates the same 1-element `uint64` zeros array.

This is trivially cacheable in two stronger ways:

1. **Pre-allocate** a single `mask` scratch buffer once in `__init__` and re-zero/re-use it.
2. **Batch the encryption.** The real win: rewrite the call site so that `_generate_masks_numpy` is invoked **once per vector** (with `n_values = L`), not `L` times with `n_values = 1`. Then each `rng.integers(size=L)` amortises its Python-level call overhead across `L` mask values, which is the whole point of the vectorised numpy API.

### (d) Call-count scaling

**`_generate_masks_numpy` is called once per (client, vector_element)**, not per (client_pair, …). The peer/pair iteration happens *inside* `_generate_masks_numpy` via the two `for rng in self._pos_rng` / `self._neg_rng` loops on lines 130 and 132.

So per round, summed across all clients, the call counts are:

| Quantity | Count | Where |
|---|---|---|
| `_generate_masks_numpy(1)` invocations | `N · L` | one per `encrypt_uint` call, one `encrypt_uint` per vector element |
| `rng.integers(size=1)` calls | `N · L · (N − 1)` | `(N−1)` peer RNGs iterated inside each `_generate_masks_numpy(1)` |
| `np.zeros(shape=(1,))` allocations | `N · L` | line 128 |

Where `N` = number of clients and `L` = number of scalars in the encrypted vector (≈ total parameter count of the model + any extra fields in the `Aggregate`).

So **`_generate_masks_numpy` itself scales O(N · L)** — linear in N. The work *inside* a single call is O(N) (the peer-RNG loop). Total work in mask generation is therefore **O(N² · L) per round**, which is consistent with the quadratic curve you observed in the speedscope profile but is **not** higher-than-quadratic (no element-level outer loop). Good — your hypothesis holds.

For your N=10 / L≈ MNIST-CNN-param-count run, this is roughly:
- `_generate_masks_numpy` calls per round: 10 · L
- `rng.integers` calls per round: 10 · L · 9 = 90 · L
- `np.zeros` allocations per round: 10 · L

That is the budget py-spy is showing you 94 % of wall time spent in.

## 5. RNG handling

The PRNGs are constructed **once** at `MaskingEncrypter.__init__` time and reused across all subsequent `_generate_masks_*` calls.

**File:** `declearn/secagg/masking/_encrypt.py`
**Lines:** 81–87

```python
81        # Set up random number generators from input seeds.
82        self._pos_rng = [
83            np.random.default_rng(seed) for seed in pos_masks_seeds
84        ]
85        self._neg_rng = [
86            np.random.default_rng(seed) for seed in neg_masks_seeds
87        ]
```

These `Generator` instances live on the encrypter for its full lifetime; their internal PCG64 state is advanced (not re-seeded) on every `rng.integers(...)` call inside `_generate_masks_numpy`. So:

- **No per-call RNG creation cost.** That hidden cost does not exist here. ✓
- The list `self._pos_rng` (and `self._neg_rng`) is iterated by `for rng in …` on lines 130/132. That iteration is cheap; what's expensive is what happens inside.
- Because each scalar mask draws a fresh `n_values=1` array, the RNG produces *one uint64 of randomness per call* despite being capable of producing thousands per call at near-identical fixed cost. The wasted potential here is the entire performance story.

## Summary — what the profile is really telling you

The 94 %-of-wall hotspot is **not** PRNG entropy generation, **not** key agreement, **not** modular arithmetic. It is **Python-level call overhead** to numpy's vectorised functions, called `O(N · L)` times per round with `size=1` each time.

If you wanted to act on this (separate decision; you only asked for investigation), the actionable target would be `encrypt_vector` in `declearn/secagg/api/_encrypt.py:157–178` — replace the per-element list comprehension on line 177 with a single batched mask draw of length `len(int_val)`. That change alone would collapse the `N · L` hot-path calls to `N` calls and convert the `(N−1)` inner-loop work from "1 mask × `(N−1)` peers" to "`L` masks × `(N−1)` peers" — same total entropy generated, vastly less Python/numpy dispatch overhead. Expected effect: the curve flattens from O(N² · L) total wall to something much closer to O(N² + N · L) effective wall, with a large constant-factor cut.

But again — investigation only. No code was changed.
