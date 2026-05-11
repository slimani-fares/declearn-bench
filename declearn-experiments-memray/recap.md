# Memray sweep + deep-dives — Recap

## Date
2026-05-11

## What this was
First memory-profiling work in the declearn loop (prior 12 experiments were
all py-spy / CPU-time). Two phases:
1. **Sweep**: memray-profile 5 representative declearn configurations to
   identify allocation hotspots.
2. **Deep-dives**: pick the top 3 most actionable findings, build patches,
   memray-A/B them to validate or refute.

## Phase 1 — Sweep results

| Tag                | Peak (MB) | Total alloc | N allocations | Wall (s) |
|--------------------|-----------|-------------|----------------|-----------|
| M01_vanilla        | 409.0     | 2.56 GB     | 1.19 M          | ~22       |
| M02_dp             | 584.8     | 3.50 GB     | 1.43 M          | ~25       |
| M03_secagg_n20     | 703.8     | 8.16 GB     | 1.65 M          | ~58       |
| M04_fairgrad       | 576.7     | 11.26 GB    | 1.66 M          | ~25       |
| M05_vector_big     | 448.6     | 38.97 GB    | 3.30 M          | ~21       |

**Three findings worth deep-diving** (per `sweep_summary.md`):
- M04 fairgrad: `_conv_forward` cumulative 4.84 GB = 8.4× vanilla. Hypothesis:
  per-group iteration causes 4× redundant conv forwards.
- M03 secagg N=20: `iterencode` (733 MB) + `raw_decode` (453 MB) = 1.18 GB
  of JSON allocation churn on the encrypted-vector wire format.
- M02 DP: `_poisson_sampling` 216 MB cumulative — declearn allocates a
  fresh (n_batches, n_samples) float buffer per call.

Setup caveats applicable to all:
- memray Python-only mode (no `--native`).
- Single-seed runs (memray output is deterministic per inputs, so this is
  fine for *pattern* identification; not enough for wall-clock claims).
- "Peak" doesn't perfectly track RSS — PyTorch's caching allocator can
  hide what's freed-to-pool-but-not-OS.
- All configs use 1 round + few SGD steps (just enough to populate heap).

## Phase 2 — Deep-dives

### DD1 — FairGrad single-pass memory (refuted)

**Hypothesis:** the single-pass `compute_groupwise_metrics` from exp_11
would reduce conv_forward allocations by K-fold (K = number of sensitive
groups) versus the canonical per-group iteration.

**Method:** memray-A/B `declearn-for-exp_11_fairness_singlepass/{master,exp_11_fairness_singlepass_variant}` on the M04 FairGrad config.

**Result:**

| Metric | Master | Variant | Δ |
|---|---|---|---|
| Peak (MB) | 576.8 | 570.7 | -6.1 (-1.1%) |
| Total alloc | 11.27 GB | 11.27 GB | ≈ 0 |
| `_conv_forward` cumul | 4.838 GB | 4.827 GB | -11 MB (-0.2%) |
| `_conv_forward` n_allocs | 429,963 | 418,957 | -11,006 (-2.6%) |

**Status: refuted.** Same root cause as exp_11's wall-clock null result —
per-group iteration and single-pass process the same *total samples*, so
total `compute_batch_predictions` forward FLOPs and resulting conv_forward
allocations are equivalent. The "K-fold redundancy" was a misread of the
call graph; the only saving is per-group iterator setup overhead which is
microsecond-scale.

### DD2 — SecAgg binary-packed JSON (confirmed)

**Hypothesis:** `MaskedAggregate.encrypted` is a `List[int]` of uint64 values
that the canonical `to_dict` emits as a Python list, which `json.dumps`
then stringifies one int at a time. Packing those values into base64-
encoded uint64 bytes would replace O(L) str allocations per vector with
O(1).

**Patch** (`declearn-for-exp_12_secagg_aggregate/dd2_json_binary_variant`,
50 lines, one file `declearn/secagg/masking/_aggregate.py`):
- `to_dict`: when `max_int <= 2**64`, pack `encrypted` into
  `array.array('Q', encrypted).tobytes()` → base64 → ascii str.
  Emit as `encrypted_b64` field; drop `encrypted`.
- `from_dict`: detect `encrypted_b64`, decode, hand off to parent.
- Wire-format change handled symmetrically on both sides — fine because
  client and server run in the same process under quickrun. Backward-
  compatible if either field is present.

**Smoke:**
- Unit round-trip: packed→unpacked identity on `[1, 2, 3, 2^63, 2^64-1]`.
- Wire-size compression at L=10,000: 213,956 chars → 106,668 chars (2×).

**Memray A/B** (M03 config, N=20, 1 round, 5 SGD steps):

| Metric | Master | DD2 variant | Δ |
|---|---|---|---|
| Peak (MB) | 703.8 | **641.9** | **-62 (-8.8%)** |
| Total alloc | 8.16 GB | **7.64 GB** | **-518 MB (-6.4%)** |
| N allocations | 1,652,585 | 1,646,720 | -5,865 (-0.4%) |
| `iterencode` cumul | 733 MB | **489 MB** | **-244 MB (-33%)** |
| `raw_decode` cumul | 453 MB | **(out of top-5)** | **eliminated** |
| Wall-clock (s)     | 58.2  | 57.9     | -0.3 (within noise) |

**Status: confirmed (memory win, wall-clock neutral).** The encrypted-
vector payload is now O(1) JSON tokens per vector instead of O(L). Peak
heap drops ~9%, cumulative allocation drops ~6%, JSON-parser allocations
basically disappear. No wall-clock cost.

**Why peak drops too** (not just cumulative): at N=20 the server holds
~20 deserialized payloads simultaneously during aggregation. Each was a
huge Python list of ints; now each is a single bytes blob.

### DD3 — DP Poisson buffer reuse (refuted — peak regression)

**Hypothesis:** declearn's `_poisson_sampling` allocates a fresh
`(n_batches, n_samples)` float64 array per call (~96 MB at the M02 default).
Caching the buffer should eliminate the cumulative allocation.

**Patch** (`declearn-for-exp_04_dp/dd3_poisson_buffer_variant`,
single file `declearn/dataset/_inmemory.py`):
- Hold `self._poisson_buf` and reuse if shape matches.
- Fill in-place via `Generator.random(out=buf)`.

**Smoke:** unit-verified that consecutive calls produce *different* draws
(RNG state advances) at the correct sample rate (0.002 ≈ 48/24000).

**Memray A/B** (M02 config, 1 round, 10 SGD steps):

| Metric | Master | DD3 variant | Δ |
|---|---|---|---|
| Peak (MB) | 584.8 | **710.0** | **+125 (+21%) WORSE** |
| Total alloc | 3.50 GB | 3.51 GB | ≈ 0 |
| `_poisson_sampling` cumul | 216 MB | 192 MB | -24 (-11%) |

**Status: refuted (peak regression).** The patch swaps a cheap transient
allocation for a long-lived retained one. The 96 MB float buffer now sits
alongside everything else at peak, inflating high-water-mark by ~125 MB
(matches buffer size + small overhead). The cumulative saving is only
24 MB. Classic memory-pooling antipattern: traded churn for peak, came
out behind.

The real lesson: PyTorch + numpy already amortize fresh allocations well
via OS-level free → page reuse. Holding a buffer to "save" the alloc
makes the high-water-mark worse, not better. Memory caching is only
useful when allocations are *small but frequent* and the allocator path
itself is the bottleneck — neither applies here.

## Cross-experiment summary

| Deep-dive | Status | Peak Δ | Cumulative Δ | Wall-clock Δ |
|-----------|--------|--------|--------------|--------------|
| DD1 (fairness single-pass) | refuted (null) | -1.1% | ~0% | ~0% (per exp_11) |
| DD2 (secagg binary JSON)   | **confirmed**  | **-8.8%** | **-6.4%** | ~0% |
| DD3 (DP poisson buffer)    | refuted (regression) | **+21%** | ~0% | not measured |

**One real win out of three.** DD2's JSON-binary-packing patch is the
output worth carrying forward. It's a 50-line single-file change with
clean backward compat and a measured memory benefit at high N that's
proportional to the encrypted-vector size.

## What we learned (process)

1. **Memory has different bottlenecks than CPU time, but also different
   *non*-bottlenecks.** The conv_forward allocation churn that dominates
   most profiles is torch's nature and not declearn-attributable. The
   declearn-attributable allocations (Vector dispatch, dataset machinery,
   secagg wire format) are smaller in absolute terms but more often
   actually fixable.
2. **"Total allocated" vs "peak" can disagree.** DD3's variant cut total
   slightly but added 125 MB of peak. The single number you optimize for
   matters: for a long-running process, peak (≈ RSS proxy) is usually
   what users feel; total is more of a churn / OS-load proxy.
3. **Memory pools only pay off in specific regimes.** Holding a buffer
   for reuse is only a win when (a) the allocation/free path itself is
   expensive (it isn't, in modern numpy/torch), or (b) the allocation
   would force the OS to grow heap and rarely shrink (PyTorch already
   handles this via its caching allocator). Otherwise you just inflate
   peak.
4. **One sweep + N deep-dives is the right ratio.** Lower-overhead than
   running A/B for every hypothesis; surfaces the load-bearing items
   without committing to a patch design ahead of time.
5. **memray `tree` command hangs under capture_output.** Don't pipe it.
   Use `memray stats` (always fine) and `memray summary -r N` (works
   with `-r`, not `--rows`). One hard-learned config detail.

## Where the data lives

| if you want…                          | open…                                                  |
|---------------------------------------|--------------------------------------------------------|
| Cross-experiment numbers (this file)  | this file                                              |
| Phase 1 sweep numbers                 | `sweep_summary.md` + `exp_M0X_*/runs/`                 |
| DD1 raw outputs                       | `exp_M04_fairgrad/dd1_runs/{dd1_master,dd1_variant}/`  |
| DD2 raw outputs                       | `exp_M03_secagg_n20/dd2_runs/dd2_variant/`             |
| DD3 raw outputs                       | `exp_M02_dp/dd3_runs/dd3_variant/`                     |
| DD2 patch                             | `declearn-for-exp_12_secagg_aggregate` branch `dd2_json_binary_variant` |
| DD3 patch                             | `declearn-for-exp_04_dp` branch `dd3_poisson_buffer_variant` |
| Memray driver                         | `_setup/run_memray.py` + `_setup/run_sweep.py`         |
| Flamegraphs                           | each `runs/<tag>/<ts>/flamegraph.html`                 |
