# SecAgg masking — decryption-side investigation

Read-only investigation. No source files were modified.
Date: 2026-04-30.

All paths below are relative to `~/work/declearn-bench/declearn-for-secagg/`.

---

## 1. Generic `decrypt_vector` source

**File:** `declearn/secagg/api/_decrypt.py`
**Lines:** 169–192

```python
169    def decrypt_vector(
170        self,
171        values: List[int],
172        specs: VectorSpec,
173    ) -> Vector:
174        """Decrypt an encrypted sum of private Vector of values.
...
188        """
189        int_val = [self.decrypt_uint(val) for val in values]
190        int_val = [self._correct_quantized_sum(val) for val in int_val]
191        flt_val = self.quantizer.unquantize_list(int_val)
192        return Vector.build_from_specs(flt_val, specs)
```

The summation/aggregation step (`sum_encrypted`) is declared abstract here at lines 73–89:

```python
73    @abc.abstractmethod
74    def sum_encrypted(
75        self,
76        values: List[int],
77    ) -> int:
78        """Sum some encrypted integer values into a single one.
```

— but note this `sum_encrypted` has a **different role** than the encrypt-side mask draw. It is called only on **small fixed-size lists** (e.g. `decrypter.sum_encrypted([self.n_steps, other.n_steps])` in `messaging.py:187,251,252,297,338` and the FairFed delta in `fairness/fairfed/_messages.py:102`). It is **not** in the per-element hot path for vectors.

The analogous per-element scalar adder for vector aggregation lives on the `MaskedAggregate` side (see §3(c) below), not on the `Decrypter`.

## 2. `MaskingDecrypter` — masking-specific Decrypter methods

**File:** `declearn/secagg/masking/_decrypt.py`
**Lines:** 35–110 (full class)

The class overrides exactly four methods, none of them perform per-element numpy work:

```python
57    secure_aggregate_cls = MaskedAggregate
59    def __init__(...) -> None:
        # records max_int = 2**bitsize and adjusts quantizer bitsize
        # for the (n-1)·q(0) shift correction; no per-element work.

88    def sum_encrypted(
89        self,
90        values: List[int],
91    ) -> int:
92        return sum(values) % self.max_int

94    def decrypt_uint(
95        self,
96        value: int,
97    ) -> int:
98        return value % self.max_int

100    def decrypt_aggregate(
101        self,
102        value: SecureAggregate[AggregateT],
103    ) -> AggregateT:
        # type/version checks then delegates to super().decrypt_aggregate(value)
```

There is **no `_unmask_numpy`**, no PRNG, no numpy ufunc on the masking decrypt side. That is by design: in the masking SecAgg variant, masks cancel out *during* aggregation (each pos-mask in client `i`'s contribution is matched by a neg-mask in client `j`'s contribution; their modular sum recovers the cleartext sum). So the "decrypt" step is trivially `value % max_int` per scalar — a single Python int op.

The `MaskingDecrypter.decrypt_aggregate` defers to the base-class `Decrypter.decrypt_aggregate` (api/_decrypt.py:194), which iterates through `value.enc_specs` and dispatches per-field via `_decrypt_value` → `decrypt_vector` / `decrypt_numpy_array` / `decrypt_float`.

## 3. Specific answers

### (a) Per-element list comprehensions on the decrypt side?

**Yes — but smaller scope and far cheaper than the encrypt side.**

In `api/_decrypt.py:decrypt_vector`, three per-element passes over the vector:

```python
189    int_val = [self.decrypt_uint(val) for val in values]                    # L iterations
190    int_val = [self._correct_quantized_sum(val) for val in int_val]         # L iterations
191    flt_val = self.quantizer.unquantize_list(int_val)                       # L iterations (inside Quantizer)
```

For the masking variant, `decrypt_uint` (line 98) is just `value % self.max_int` and `_correct_quantized_sum` (line 118) is just `value - self._qt_corr`. Each is a single Python-int op. No numpy allocation, no ufunc dispatch, no PRNG. The cost is dominated by attribute lookup and Python-level list construction, not by computation.

`decrypt_numpy_array` (lines 140–167) has the same shape — same two list comprehensions plus an extra `[round(x) for x in s_val]` if the dtype is signed integer.

### (b) Call counts per (round, server-side aggregate)

Per round, on the server, for one encrypted aggregate field of length `L`:

| Call | Count per round | Why |
|---|---|---|
| `decrypter.decrypt_aggregate(...)` | **1** per encrypted aggregate | called once after all client contributions are summed (`messaging.py:169,171,239`) |
| `decrypt_vector(values, specs)` | **1** per `Vector` field of that aggregate | dispatched from `_decrypt_value` (`api/_decrypt.py:265`) |
| `decrypt_uint(val)` | **L** | line 189 list comprehension |
| `_correct_quantized_sum(val)` | **L** | line 190 list comprehension |
| `quantizer.unquantize_list(...)` | **L** internally | line 191 |

So total per-element decrypt ops per round per vector field ≈ **3 · L** — and crucially **independent of N**.

This is **not** a quadratic hot path. The encrypt-side antipattern produces `O(N² · L)` PRNG calls per round (see prior investigation report); the decrypt path produces `O(L)` Python-int ops per round, total. **Decryption is not the problem.**

### (b ′) Where N really shows up on the server side: `MaskedAggregate.aggregate_encrypted`

Worth flagging because it *is* a per-element loop and it *does* scale with N — just not on the `Decrypter`:

**File:** `declearn/secagg/masking/_aggregate.py`
**Lines:** 85–92

```python
85    def aggregate_encrypted(
86        self,
87        val_a: List[int],
88        val_b: List[int],
89    ) -> List[int]:
90        return [
91            (a + b) % self.max_int for a, b in zip(val_a, val_b, strict=False)
92        ]
```

Called from `api/_aggregate.py:133` (inside `SecureAggregate.aggregate`), which fires every time the server folds another client's contribution into the running aggregate. Across one round that is `(N − 1)` times, each running `L` Python int adds and modulos.

Total per round per encrypted aggregate: **`(N − 1) · L` Python int ops** in this list comprehension — i.e. **O(N · L) on the server**. Linear, not quadratic, but still a per-element Python loop where a numpy/uint64 array sum would do the same work in microseconds.

If your speedscope shows server self-time leaking into `aggregate_encrypted` at higher N, that is the second batching opportunity beyond `encrypt_vector`.

### (c) Is there an underused vectorised `_unmask_numpy`?

**No.** Greping `declearn/secagg/masking/_decrypt.py` confirms only the four methods listed in §2; no `_unmask_*` helpers exist. The architectural reason is sound — there are no masks to apply on decrypt for the masking variant; masks cancel in the modular sum. So no underused-vectorised-helper opportunity equivalent to `_generate_masks_numpy` on this side.

The closest analog to what you'd vectorise is `aggregate_encrypted` (above): you could replace the list comprehension with a `(np.asarray(val_a, dtype=uint64) + np.asarray(val_b, dtype=uint64)) % max_int` and convert back at the end — but that is a separate refactor target on `MaskedAggregate`, not on `MaskingDecrypter`.

## 4. Full call surface for `encrypt_vector`

**Defined:** `declearn/secagg/api/_encrypt.py:157`
**Called from:** exactly **one** internal site — no external callers.

```bash
$ grep -rn "encrypt_vector" declearn/ --include="*.py"
declearn/secagg/api/_encrypt.py:157:    def encrypt_vector(
declearn/secagg/api/_encrypt.py:245:            return self.encrypt_vector(value)
```

The single call site is `Encrypter._encrypt_value`:

```python
242        if isinstance(value, np.ndarray):
243            return self.encrypt_numpy_array(value)
244        if isinstance(value, Vector):
245            return self.encrypt_vector(value)
246        if isinstance(value, float):
247            return [self.encrypt_float(value)], True
```

`_encrypt_value` itself is only called from `encrypt_aggregate` (api/_encrypt.py:214 — the loop over `cryptable.items()`).

So the dependency tree is:

```
encrypt_aggregate         (api/_encrypt.py:180)
  └── _encrypt_value      (api/_encrypt.py:225)
        └── encrypt_vector(api/_encrypt.py:157)   ← single call site
```

**No other call sites exist** in the declearn tree (searched all `.py` under `declearn/`). The same is true for `decrypt_vector` — defined at `api/_decrypt.py:169`, only called from `_decrypt_value` at `api/_decrypt.py:265`, which itself is only called from `decrypt_aggregate`.

This is a small, contained refactoring surface. Anything you change in `encrypt_vector` propagates only through the `encrypt_aggregate` codepath; no callers reach into it directly. Same for `decrypt_vector`.

Worth noting for completeness: the sister methods `encrypt_numpy_array` (line 115) and `encrypt_float` (line 96) share the same single-caller property (only `_encrypt_value`). Both have the same per-element pattern as `encrypt_vector` (line 153: `[self.encrypt_uint(val) for val in int_val]`). If you refactor `encrypt_vector` to do a single batched mask draw of length `len(int_val)`, the equivalent change would also apply to `encrypt_numpy_array:153` for symmetry — and that is also where py-spy will eventually point if you encrypt mostly numpy-array-typed fields rather than Vector-typed ones.

## Summary

- **Decryption is not the analog of the encrypt-side hotspot.** The masking variant does no mask draws on decrypt; per-element ops are pure Python int modulo / subtract, and the count is `O(L)` per round, **independent of N**.
- **No vectorised `_unmask_numpy` exists** because none is needed for masking — mask cancellation is implicit in modular summation.
- **The server-side N-dependent per-element loop** lives in `MaskedAggregate.aggregate_encrypted` (`masking/_aggregate.py:85–92`), not on the `Decrypter`. It is `O(N · L)` per round and would benefit from numpy batching if it shows up in your profile, but this is a *secondary* target.
- **`encrypt_vector` has exactly one internal call site** (`api/_encrypt.py:245`, inside `_encrypt_value`, only reached via `encrypt_aggregate`). The change surface is small and contained. The same applies to `decrypt_vector` and to the sister methods `encrypt_numpy_array` / `encrypt_float`, which share the per-element antipattern at line 153.

So: act on `encrypt_vector` (and consider doing `encrypt_numpy_array:153` in the same PR for symmetry). The decrypt side does not need the same surgery.

No source files were modified.
