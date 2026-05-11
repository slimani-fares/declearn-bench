# Memray sweep — cross-experiment summary

## Date
2026-05-07

## Runs surveyed
| Tag                | Peak (MB) | Total allocated | N allocations | Wall-clock (s) |
|--------------------|-----------|-----------------|----------------|------------------|
| M01_vanilla        | 409.0     | 2.56 GB         | 1.19 M          | ~22              |
| M02_dp             | 584.8     | 3.50 GB         | 1.43 M          | ~25              |
| M03_secagg_n20     | 703.8     | 8.16 GB         | 1.65 M          | ~58              |
| M04_fairgrad       | 576.7     | 11.26 GB        | 1.66 M          | ~25              |
| M05_vector_big     | 448.6     | 38.97 GB        | 3.30 M          | ~21              |

All runs at 1 round, ≤10 SGD steps. Memray Python-only mode (no `--native`).

## Top allocators per experiment (by cumulative bytes)

### M01 — vanilla baseline
1. `_conv_forward` (torch) — 575 MB
2. `relu` (torch) — 284 MB
3. `_max_pool2d` (torch) — 210 MB
4. `read_array` (numpy data loading) — 188 MB
5. `dedent` (Python textwrap, in import-time docstring processing) — 124 MB

→ heap is dominated by torch's own conv-forward allocation pattern; nothing
declearn-attributable above the noise floor.

### M02 — DP-SGD
1. `_conv_forward` — 586 MB
2. `_engine_run_backward` (autograd) — 420 MB **(NEW vs vanilla)**
3. `relu` — 284 MB
4. **`_poisson_sampling`** (`declearn/dataset/_inmemory.py:574`) — **216 MB**
5. `_max_pool2d` — 210 MB

→ the autograd backward heap (420 MB) is the vmap per-sample-gradient cost,
expected. **`_poisson_sampling` at 216 MB is declearn-side, novel** — every
DP step constructs a fresh boolean Poisson mask.

### M03 — SecAgg masking N=20
1. `_conv_forward` — 750 MB
2. **`iterencode`** (Python json/encoder.py) — **733 MB**
3. **`raw_decode`** (Python json/decoder.py) — **453 MB**
4. **`_apply_operation`** (Vector base) — 398 MB
5. `relu` — 368 MB

→ **JSON encode + decode account for 1.18 GB cumulative** of allocation per
round at N=20 — that's the encrypted-vector payloads being stringified
(server side: receive 20 client replies, parse each from JSON; client side:
encode encrypted reply as JSON before send). At N=20 with L≈10k uint64
values per vector, this is huge. **Clear declearn-side optimization
opportunity.**

### M04 — FairGrad
1. **`_conv_forward`** — **4.84 GB** (8.4× vanilla)
2. **`relu`** — **2.40 GB** (8.4× vanilla)
3. **`_max_pool2d`** — **1.78 GB** (8.5× vanilla)
4. `_build_iterator` (declearn dataset) — 346 MB
5. `read_array` — 189 MB

→ **the per-group iteration in `compute_groupwise_metrics` causes ~8×
the conv-forward allocation churn vs vanilla**. With K=4 sensitive groups,
each iterated separately over its group's training subset, the fairness
round drives conv_forward 4× more times than necessary. This is the
*memory-side* of the same redundancy exp_11's single-pass patch
addressed (which was a wall-clock null but might be a memory win).

### M05 — BiggerCNN with foreach variant
1. `_conv_forward` — **20.45 GB**
2. `relu` — **6.83 GB**
3. `batch_norm` — **6.51 GB**
4. `_engine_run_backward` — 4.02 GB
5. `read_array` — 188 MB
6. `_apply_operation` (in `model/torch/_vector.py:151` — the variant override) — 206k allocations

→ heap totally dominated by torch ops on a 20-conv-layer model — expected.
The `_apply_operation` Python-side allocation count (206k) suggests the
foreach variant DOES add transient allocations (key lists, tensor lists,
dict reconstructions per call). Compared to canonical exp_10 master arm
this might show extra churn — worth a paired memray A/B.

## Top 3 deep-dive candidates

1. **DD1 — FairGrad single-pass memory**: at M04 the per-group iteration
   drives conv_forward to 4.84 GB cumulative (8.4× vanilla). Repurpose
   exp_11's already-written single-pass patch and measure the memory
   delta — even though wall-clock didn't move, the allocation churn might.
   Predicted: ~3-4 GB cumulative reduction, ~K-fold reduction in conv
   allocations.

2. **DD2 — SecAgg JSON serialization at high N**: M03 shows iterencode +
   raw_decode totaling 1.18 GB cumulative at N=20. The encrypted payload
   is `List[int]` serialized as JSON. Switching to a compact binary
   encoding (one `bytes` blob per vector via `int.to_bytes`/`from_bytes`)
   should eliminate per-int stringification. Predicted: 90%+ reduction
   in JSON-related allocations on encrypted-vector payloads.

3. **DD3 — DP per-step Poisson mask reuse**: M02 shows `_poisson_sampling`
   at 216 MB cumulative. Each step builds a fresh boolean mask of length
   `n_samples`. With pre-allocation/reuse this cost goes to zero.
   Predicted: 200+ MB cumulative reduction per run.

## Method caveats
- Single-seed runs. Memray output is deterministic given the same inputs,
  so single-seed is fine for *allocation pattern* identification, but
  for any wall-clock claim it's not enough.
- "Total allocated" is cumulative through the run — high values indicate
  churn (allocate, free, allocate, free), not high peak.
- "Peak" is the high-water mark of simultaneously-live bytes; this is
  closer to RSS but doesn't perfectly match the OS view (PyTorch's
  caching allocator hides what's freed-to-pool-but-not-OS).
- All experiments use the same small mnist-quickrun data baseline except
  M05 (which uses the bigger CNN from exp_10).

## Where the data lives

| if you want…                          | open…                                                  |
|---------------------------------------|--------------------------------------------------------|
| Run-level stats (peak/total/n_allocs) | `exp_M0X_*/runs/<tag>/<ts>/summary.txt`                |
| Visual flame                          | `exp_M0X_*/runs/<tag>/<ts>/flamegraph.html`            |
| Run metadata                          | `exp_M0X_*/runs/<tag>/<ts>/metadata.json`              |
| Cross-experiment numbers (this file)  | this file                                              |
| Sweep driver                          | `_setup/run_memray.py`, `_setup/run_sweep.py`          |
