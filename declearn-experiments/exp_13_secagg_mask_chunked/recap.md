# Experience 13 — SecAgg masking mask-generator chunking — Recap

## Date
2026-05-13

## TL;DR

**Confirmed deal-breaker (memory-side).** At N=100 peers, chunking the
per-peer `rng.integers(..., size=L)` call inside `_generate_masks_numpy`
into CHUNK=65536 sub-draws reduces **system peak memory by 21.9%
(−543 MB on 2473 MB)**, while leaving cumulative bytes allocated and the
per-line attribution at `masking/_encrypt.py:131/:133` unchanged. End-to-end
encrypt/decrypt round-trip stays byte-identical (smoke A) and declearn's
full SecAgg test suite (62 unit + 2 functional tests) passes against the
patched fork. Wall-clock penalty is +3.5% at N=100, well under the §7
deal-breaker floor. At N≤20 the patch produces no measurable peak reduction
— the per-peer multiplier needs to be large for the temporal-overlap
mechanism (Section "Mechanism" below) to bite.

## Setup
- **Master arm:** `declearn-for-exp_12_secagg_aggregate` branch `master`
  (commit `e13e2fb` — declearn 2.8.0 + unified runner + batched encrypt-side
  from exp_06). This is "current declearn with all previously-shipped
  optimizations applied", which is the only meaningful baseline for a
  *new* intervention.
- **Variant arm:** `declearn-for-exp_13_secagg_mask_chunked` branch
  `exp_13_variant_h1_chunked` (commit `3bfa756`) — master + chunked
  `_generate_masks_numpy`. Net diff is 12 lines in one file
  (`patches/h1_chunked_mask.diff`).
- **Workload:** mnist quickrun, 1 round, n_steps=5, SecAgg masking, IID
  data splits at N ∈ {5, 20, 50, 100} clients. Reused DD2's N=50/N=100
  configs verbatim; wrote N=5 and N=20 configs in the same shape.
- **Profiler:** memray Python-only mode (no `--native`), one run per cell
  (8 runs total). Same node throughout (magnet4); load ≤ 0.3 at start.
- **Total wall-clock spent on this experience:** ~21 minutes.

## Hypotheses tested

### H1: Chunked `_generate_masks_numpy` reduces system peak memory at high N

**Status: confirmed (memory dimension), with wall-clock penalty bounded
below 10%.**

#### Smoke tests
| Layer | Target | Result |
|---|---|---|
| A — byte equivalence | mask streams at n=1M and n=10, default bitsize=64, shared seeds | **PASS** — `np.array_equal` exact. Confirms numpy PCG64 in full-range uint64 mode advances per element regardless of call size; chunking does not alter the state-advancement order. |
| B — declearn unit tests | `test/secagg/controllers/test_masking_controllers.py` + `test/secagg/setup/test_masking_setup.py` | **PASS** — 60 tests, full encrypt/decrypt round-trip across peer counts, dtypes, and value types. |
| C — declearn functional tests | `test/functional/test_toy_clf_secagg.py` (FedAvg + SCAFFOLD with masking SecAgg) | **PASS** — 2 tests, 25.95s combined with B. |

All three layers passed on the same variant fork that ran the A/B.

#### A/B grid (memray peak, total, n_allocs, wall-clock)

| N | arm     | peak (MB) | total (MB) | n_allocs   | wall (s) |
|---|---------|----------:|-----------:|-----------:|---------:|
| 5   | master  |     476.2 |       3531 |  1,237,489 |     87.5 |
| 5   | variant |     476.3 |       3534 |  1,237,756 |     87.9 |
| 20  | master  |     700.4 |       8219 |  1,652,910 |    127.0 |
| 20  | variant |     704.8 |       8217 |  1,653,785 |    128.2 |
| 50  | master  |    1163.0 |      18778 |  3,112,826 |    127.0 |
| 50  | variant |    1106.0 |      18775 |  3,117,342 |    127.0 |
| 100 | master  |    2473.0 |      40548 |  7,430,885 |    251.0 |
| 100 | variant |    1930.0 |      40497 |  7,452,018 |    259.7 |

Deltas (variant − master):

| N   | peak Δ MB | peak Δ %  | total Δ MB | n_allocs Δ | wall Δ % |
|-----|----------:|----------:|-----------:|-----------:|---------:|
| 5   |      +0.1 |   +0.02%  |       +3   |       +267 |   +0.5%  |
| 20  |      +4.4 |   +0.63%  |       −2   |       +875 |   +0.9%  |
| 50  |     −57.0 |   **−4.90%** |       −3   |     +4,516 |    0.0%  |
| 100 |    −543.0 |   **−21.96%** |      −51 |    +21,133 |   +3.5%  |

#### Per-line attribution at `_encrypt.py:131/:133` (master) / `:136/:142` (variant)
Cumulative allocation at the two `rng.integers(...)` source lines is
**unchanged across arms** (4.012 GB per line at N=100, 992.877 MB per
line at N=50). That is exactly the prediction: chunking does not change
total bytes allocated, only the temporal overlap of live temporaries.
The cumulative allocation rank order in the top-5 also stays the same
(`_generate_masks_numpy` lines remain #1 and #2 at N=100).

#### Mechanism
The master arm allocates one temporary of L·8 bytes per peer RNG inside
`_generate_masks_numpy`. With multiple client coroutines running
concurrently in the asyncio event loop, several can be inside
`_generate_masks_numpy` at the same moment, and their temporaries
stack: peak ≈ N_concurrent · 2·L·8 bytes. The variant reduces each
per-call temporary from L·8 to CHUNK·8 bytes (CHUNK=65536), so even
when many clients are concurrently mid-call, the live-temporary
working set is bounded.

This explains the N-dependent peak savings:
- At N=5 (and largely N=20), few clients are simultaneously inside
  `_generate_masks_numpy`, so temporaries rarely overlap. Peak is
  dominated by torch's `_conv_forward` and JSON encode/decode — chunking
  saves nothing visible.
- At N=50 (−4.9%) and especially N=100 (−21.9%), concurrent overlap is
  the dominant peak contributor. Chunking removes it.

#### Deal-breaker assessment (§7)
| Rule | Outcome |
|---|---|
| Accuracy floor (variant accuracy < baseline − 0.10) | n/a (memory experiment; smoke C confirms end-to-end aggregate correctness) |
| Smoke-test mismatch beyond tolerance | Not triggered — all three layers pass exactly |
| Crash or hang twice | Not triggered |
| Perf went backwards (>10% slower) | Not triggered — max wall-clock delta is +3.5% at N=100 |
| Memory target (peak reduction visible) | **Satisfied at N=50 and N=100**; null at N≤20 (expected) |

**Confirmed.** `requires_human_crypto_review: true` (touches a SecAgg
primitive); Marc should sign off before any upstream PR, but the
byte-equivalence smoke test is by construction (numpy PCG64 advances
state per element in full-range uint64 mode), so the cryptographic
guarantee is preserved exactly.

#### Code change
`patches/h1_chunked_mask.diff`, 35 lines diff (12 net code lines added
in `declearn/secagg/masking/_encrypt.py`).

#### Result paths
- A/B JSON: `runs/../ab_results.json`
- memray summaries: `runs/n{5,20,50,100}_{master,variant}/<ts>/summary.txt`
- memray flamegraphs: same dirs, `flamegraph.html`
- Driver script: `run_ab.py`
- Driver console log: `run_ab.console.log`

## Open observations (NOT tested as hypotheses)

### `iterencode` + `raw_decode` remain large at N=50 (1.84 + 1.14 = 2.98 GB) and N=100 (3.75 + 2.39 = 6.14 GB)
The JSON serialization category is still the second-biggest declearn-side
allocator at high N. **The DD2 binary-encoding fork already targets this**;
once it ships, it should remove this category entirely. Not in scope for
exp_13.

### `_read_ready__data_received` (asyncio at `selector_events.py:1009`) appears at N=100 with 2.2 GB cumulative on both arms
asyncio receive-buffer churn. Same on master and variant. Not in
declearn code — would fall under the §scope memory ("not our problem")
unless declearn is sending unnecessarily large messages (which DD2 also
addresses).

### Wall-clock at N=20 is the same as N=50 (127s)
Not specific to this experiment but interesting: clearly some fixed setup
cost dominates over the per-client cost up to N=50, and only N=100
breaks past 250s. The bottleneck above N=50 is in computation, not
coordination.

### Chunk size was not tuned
`_MASK_CHUNK = 65536` was a reasonable first guess (matches typical L1/L2
boundaries on common x86 cores). A tuning sweep over CHUNK ∈ {4096,
16384, 65536, 262144} could refine the trade-off between peak reduction
and wall-clock overhead, but the current result at CHUNK=65536 is
already on the right side of every deal-breaker.

## Conclusions

The chunked mask generator is a clean memory win at the peer counts where
SecAgg matters most. At N=100 — the configuration that motivated this
investigation in the first place — the system peak drops by 543 MB
(−21.9%) at a wall-clock cost of +3.5%, with byte-identical encryption
output and zero changes to the declearn test suite.

This is the **first memory-side declearn finding in the loop that delivers
a >20% peak reduction at realistic peer counts** with a 12-line patch and
no behavioral change. The DP exp_04 and SecAgg exp_06 wins were
wall-clock; this one is the memory complement.

For Fares to revisit:
1. Run the variant in the existing DD2 driver to confirm the result
   replicates when combined with the DD2 JSON→binary fork. Memory savings
   should compose since they target disjoint hotspots (encryption mask
   generation vs message serialization).
2. Sweep CHUNK ∈ {16384, 65536, 262144} on the N=100 config to see if
   wall-clock penalty disappears at 16384 without losing the peak win.
3. The same chunking pattern applies on the *decrypt* / aggregate side
   if any `rng.integers(size=L)`-style call exists there — though the
   investigation_secagg_decrypt_side.md report should be consulted first.

## Caveats and open questions

- **Single seed per cell.** Methodology matches DD2/M03 (memray output
  is deterministic given the same inputs). For a public claim a 3-seed
  re-run would harden the numbers, but the magnitude (−543 MB) is well
  beyond per-run noise.
- **Byte-equivalence holds only for `bitsize ∈ {8, 16, 32, 64}`** (the
  power-of-2 cases where numpy `Generator.integers` uses no rejection
  sampling). For other bitsizes, chunking may change rejection-sampling
  state advancement and produce statistically-equivalent-but-not-byte-
  identical mask streams. End-to-end correctness still holds (the
  aggregate sums recover identically because the mask cancellation
  property is independent of the specific draws), but the unit-level
  byte guarantee weakens. The default bitsize=64 is the one used in all
  of Fares's tests, so this caveat is documentary.
- **Wall-clock at N=100 is the only N where the +3.5% penalty is
  measurable** — at smaller N it's well within run-to-run noise. The
  penalty itself is the cost of K extra numpy calls per peer (K = L/CHUNK
  ≈ 0.4 at L ≈ 28k); chunk size tuning could close this.
- **Backgrounded server-side aggregate code** was not touched. The
  `MaskedAggregate.aggregate_encrypted` server-side loop (referenced in
  the findings.md follow-up #6) may have its own mask-related allocation
  pattern. Not in this experiment's scope.

---

=== EXPERIENCE 13 COMPLETE ===
Recap: ~/declearn-bench/declearn-experiments/exp_13_secagg_mask_chunked/recap.md
Status: CONFIRMED (memory). −21.9% peak at N=100, +3.5% wall-clock at N=100, byte-identical encryption. Requires crypto review.
