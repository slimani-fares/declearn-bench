# Experience 12 — SecAgg masking aggregate-side vectorization — Recap

## Date
2026-05-07

## What this experiment was
The exp_06 follow-up (validated `declearn-for-secagg-batched`) closed
the encrypt-side hotspot for masking secagg. The hypothesis going into
exp_12: the **aggregate-side** loop in
`MaskedAggregate.aggregate_encrypted` (`declearn/secagg/masking/_aggregate.py:85-92`)
— a Python list-comprehension `(a + b) % max_int` over all encrypted
scalars, called (N-1) times per round to merge N client vectors —
becomes the new bottleneck at higher client counts.

This experiment tested whether vectorizing that loop with numpy
`uint64` arithmetic produces a measurable wall-clock speedup at
N=5/10/20 with masking secagg enabled.

## What changed
- **Fork:** `declearn-for-exp_12_secagg_aggregate/`, branched from
  `declearn-for-secagg-batched/` (which already contains the encrypt-
  side fix: `Encrypter.encrypt_uint_vector` calling
  `_generate_masks_numpy(L)` in one shot rather than once per scalar).
  - `master` arm: secagg-batched as-is. Encrypt-side already vectorized,
    aggregate-side still the per-element Python loop.
  - `variant` arm: + aggregate-side patch.
- **Patch:** `patches/aggregate_uint64.diff` (50 lines, single file:
  `declearn/secagg/masking/_aggregate.py`).
  - When `max_int == 2**64` (the default): cast both vectors to numpy
    `uint64`; `arr_a + arr_b` produces the natural wrap-around which
    is exactly `(a + b) % 2**64`.
  - When `max_int <= 2**63`: explicit `((arr_a + arr_b) % mod)` on
    `uint64` (the sum fits without overflow before modulo).
  - When `2**63 < max_int < 2**64` or `max_int > 2**64`: fall through
    to the canonical Python loop. Defensive.

## Hypothesis tested

### H1 — Vectorizing aggregate_encrypted produces a meaningful wall-clock speedup at higher N

**Status: refuted.**

#### Smoke

**Unit-level byte-equality** (synthetic random `uint64` vectors, both
default and non-default `max_int`):

| max_int | output equality | master timing | variant timing | speedup |
|---------|-----------------|---------------|----------------|---------|
| `2**64` | byte-equal      | 7.7 ms (L=50k) | 6.3 ms        | 1.22×   |
| `2**32` | byte-equal      | 7.4 ms (L=50k) | 6.0 ms        | 1.23×   |

**Function-level micro-benchmark** across L (single-call latency,
mean of 10 reps):

| L          | master (ms) | variant (ms) | speedup |
|------------|-------------|--------------|---------|
| 1,000      | 0.16        | 0.11         | 1.45×   |
| 10,000     | 1.35        | 1.03         | 1.31×   |
| 50,000     | 8.14        | 5.17         | 1.57×   |
| 100,000    | 15.88       | 11.88        | 1.34×   |
| 500,000    | 81.95       | 72.51        | 1.13×   |
| 1,000,000  | 165.77      | 144.76       | 1.15×   |

The patch does what it claims at the function level. The speedup is
modest (~1.2–1.5×) because canonical Python integer arithmetic with
list-comp is already reasonably fast for `int+mod`; the numpy version
saves the per-element interpreter dispatch but pays array-construction
overhead.

**End-to-end smoke** (1 round secagg-masking at N=5, 1 round, 5 SGD
steps): both arms succeed; final averaged loss differs by 7e-5
(relative 3e-5), well within float-reorder tolerance.

#### A/B (3 N-values × 2 arms × 3 seeds × 1 round = 18 runs)

| N  | arm     | wall (s, mean) | std  | speedup vs master | loss (mean) |
|----|---------|----------------|------|-------------------|-------------|
| 5  | master  | 44.73          | 0.56 | 1.00×             | 2.297604    |
| 5  | variant | 44.82          | 0.65 | 1.00× (+0.09 s)   | 2.296270    |
| 10 | master  | 46.50          | 0.58 | 1.00×             | 2.297410    |
| 10 | variant | 46.16          | 0.03 | 1.01× (-0.34 s)   | 2.295815    |
| 20 | master  | 51.04          | 0.21 | 1.00×             | 2.297126    |
| 20 | variant | 51.05          | 0.24 | 1.00× (+0.01 s)   | 2.295122    |

**No measurable wall-clock speedup at any N value tested.** The N=10
arm shows a +0.34 s nominal improvement (≈ 0.7%) but it's well within
the master std (0.58 s).

#### py-spy at N=20 (single-seed, 2 rounds, 100 Hz)

Profile total: master 17.78 s of self-time over 54.5 s wall-clock,
variant 18.73 s over 55.6 s wall-clock.

**Top frames at N=20 (master):**

| frame                                         | self_s | self_% |
|-----------------------------------------------|--------|--------|
| `encode` (websockets `permessage_deflate`)    | 4.70   | 26.43% |
| `send` / `write_frame` (websockets)           | 4.78   | 26.88% |
| `serialize` (websockets frames)               | 4.71   | 26.49% |
| `_call_with_frames_removed` (importlib)       | 3.92   | 22.05% |
| ...                                            | ...    | ...    |
| `aggregate_encrypted` (the patched function)  | **0.29** | **1.63%** |

**The aggregate-side cost the patch attacks is 1.6% of profile self-time
at N=20, not the dominant hotspot the hypothesis predicted.** The
dominant cost at N=20 is **websocket compression** (`permessage_deflate.encode`)
at ~30% of self-time — the more clients there are, the more frames the
server compresses.

**Variant vs master deltas** (top 25 self-time deltas, A/B comparison):

| frame                                   | master self_s | variant self_s | delta  |
|-----------------------------------------|---------------|----------------|--------|
| `aggregate_encrypted`                   | 0.29          | 0.37           | **+0.08** |
| `permessage_deflate.encode` (unrelated) | 4.71          | 5.06           | +0.35 (noise) |
| `encrypt_uint_vector` (unrelated)       | 0.33          | 0.52           | +0.19 (noise) |

`aggregate_encrypted` self-time goes UP by 0.08 s in the variant. This
looks contradictory to the unit-level 1.5× speedup, but is consistent
once you read what py-spy measures: master spends time in the Python
interpreter loop (which py-spy attributes to `aggregate_encrypted`'s
own frame), variant spends time inside numpy's C kernels (which py-spy
also attributes to the calling Python frame because numpy's C code
isn't traceable). Both are tagged identically, just with different
underlying mechanics. The variant's array-construction overhead pushes
the attribution slightly higher.

The micro-benchmark numbers are the truer measurement: variant saves
~0.05–0.08 s per round across 19 aggregate calls at L≈50k. On a 51 s
end-to-end round at N=20, that's a 0.1–0.15% improvement — far below
the wall-clock noise floor.

#### Why the hypothesis failed

The argument was: aggregate-side cost is `O(N · L)` per round, encrypt-
side stays `O(L)` per client per round. So at high N, aggregate-side
should dominate.

The arithmetic is right but the constants are wrong:
- Per-aggregate call latency ≈ 7-8 ms at L=50k for canonical, 5-6 ms
  for variant.
- 19 aggregate calls per round at N=20 → ~150 ms cumulative master,
  ~100 ms variant.
- Saved ~50 ms per round.
- Out of 51 s end-to-end wall-clock per round, 50 ms is 0.1%.
- Run-to-run wall-clock std is 200-700 ms (0.4-1.4%).

The cost path that DOES grow visibly with N is **websocket framing
and compression**: the server has to encode and send N reply frames
per client per round, and at N=20 that hits ~30% of profile time.
The aggregate-side never gets near "dominant" at any N tested.

#### Deal-breaker assessment per CLAUDE.md §7

| rule                                                       | status |
|------------------------------------------------------------|--------|
| Smoke equivalence (byte-equal at unit level)               | ✓ |
| Smoke equivalence (e2e final-loss within 1e-3 relative)    | ✓ (3e-5 actual) |
| Improvement above noise (5%)                               | ✗ — best is ~0.7% nominal at N=10, within 1 std |
| Consistency across seeds (mean improvement > 1 std)        | ✗ — at all 3 N values |
| Perf direction (variant must be faster)                    | ≈ — variant marginally faster on N=10, marginally slower on N=5/N=20, all within noise |

Multiple §7 rules failed → **refuted** (no deal-breaker triggered, but
no confirming evidence either).

## Open observations (NOT tested as hypotheses)

- **Websocket compression dominates at N=20.** `permessage_deflate.encode`
  takes ~30% of profile self-time. This is the actual scaling bottleneck
  for masking secagg at higher N — every encrypted reply gets deflate-
  compressed before being framed. On a localhost benchmark there's no
  bandwidth case for compression, just CPU cost. **This is a candidate
  for a separate experiment**: disable permessage-deflate for localhost
  runs, or only enable it above a payload-size threshold. The user-
  facing knob would be a websockets configuration option, not a
  declearn change.
- **Per-client serialize/send cost (~26-27% combined) suggests the
  SecAgg reply payload at L≈50k is large enough that JSON encoding +
  websocket framing themselves are non-trivial.** Cleartext-mode runs
  the same path so this is shared.
- **The per-aggregate-call cost in canonical (8 ms at L=50k) is not
  far from python-int + mod's theoretical minimum.** Even a perfect
  C-extension implementation would only save 5-7 ms per call, capping
  the ceiling on this optimization at ~140 ms saved per round at N=20.

## Conclusions

**Headline:** the aggregate-side patch is correct (byte-equal smoke,
e2e equivalent within float-reorder noise) but produces no measurable
wall-clock improvement at N=5, 10, or 20. The 1.5× function-level
speedup translates to ~0.1% wall-clock speedup at FL-pipeline scale —
well below the noise floor.

The framing in the original lead — "at higher N the aggregate becomes
the new dominant cost" — is empirically wrong at N=20. The actual
new dominant cost (websocket compression) is unrelated to declearn's
secagg arithmetic and grows with the number of frames sent, not with
the size of the aggregated vector.

**What this means for declearn:**
1. Do not merge this variant. The optimization is correct but the
   leverage on it is too small to justify the maintenance surface.
2. The encrypt-side fix (already in `declearn-for-secagg-batched`) is
   the better target — it dropped wall-clock from 148 s → 11 s at N=5
   per exp_06. There's no comparable opportunity on the aggregate
   side because the canonical loop is already much smaller.
3. **The next real optimization for masking secagg at scale is on the
   websocket / network side**, not the secagg arithmetic.

**What we learned (process, not declearn):**
1. **Big-O analysis without measuring constants is misleading.** The
   `O(N · L)` aggregate cost is real; what we missed was that L is
   small enough (and integer arithmetic fast enough) that the constant
   makes the absolute cost <2% of profile time even at N=20. Worth
   profiling before optimizing.
2. **Function-level speedup ≠ wall-clock speedup** when the function
   isn't on the load-bearing path. A 1.5× speedup on a function that's
   1.6% of profile time is 0.6% of wall-clock — invisible in noise.
3. **py-spy reads through the abstraction** in a useful way: when the
   patch moves work from CPython interpreter to numpy C kernel, the
   self-time attribution to the calling Python frame doesn't drop
   because both end up tagged to the same frame. Compare via
   wall-clock + micro-benchmark, not py-spy alone.

## Caveats and open questions

- **3 seeds is at the lower limit.** At ~50 s wall-clock per run with
  std ≈ 0.5 s, 3 seeds give a 95% CI on the mean of ~±0.6 s. A 1%
  effect (~0.5 s) is at the edge of detectability; more seeds would
  shrink it but the data direction (no effect) is consistent.
- **N=20 is the scale we tested; the user's lead mentioned N=100 as a
  potential regime.** At N=100 the linear-in-N aggregate cost would
  reach ~750-1500 ms saved per round, which IS detectable. But the
  websocket compression cost grows even faster (more clients, more
  frames) and would still dominate. To make N=100 measurable, use a
  smaller L (smaller model) or disable compression first.
- **The configs use `n_steps=5` per round** to keep the run small and
  expose secagg costs. With longer training (full epoch ≈ 125 steps),
  the secagg fraction shrinks to single-digit-percent of wall-clock
  regardless, so the patch matters even less.
- **`max_int = 2**64` is the only path tested e2e.** The smaller-
  `max_int` branch was unit-tested but not exercised in an FL run.

## Where the data lives

| if you want…                                | open…                                                                  |
|---------------------------------------------|------------------------------------------------------------------------|
| The 18 A/B run logs                         | `runs/ab_n*_*.log`                                                     |
| A/B summary (machine-readable)              | `runs/ab_results.json`                                                 |
| A/B summary (human)                         | `runs/ab_run.console.log`                                              |
| Master py-spy profile (N=20)                | `runs/profiles_n20/master/2026-05-07_13-38-31/pyspy_speedscope.json`   |
| Variant py-spy profile (N=20)               | `runs/profiles_n20/variant/2026-05-07_13-39-30/pyspy_speedscope.json`  |
| Side-by-side profile comparison             | `runs/compare_n20.txt`                                                 |
| The patch                                   | `patches/aggregate_uint64.diff` (50 lines, single file)                |
| Unit smoke driver + pickles                 | `smoke_unit.py` + `runs/smoke_unit_*.pkl`                              |
| End-to-end smoke (N=5)                      | `smoke_e2e.py` + `runs/smoke_e2e_*.log`                                |
| Per-N configs                               | `config_secagg_n{5,10,20}.toml`                                        |
| A/B driver                                  | `ab_run.py`                                                            |
| Profile driver                              | `profile_arms.py`                                                      |

## Status

**Refuted (per CLAUDE.md §7).** Patch is correct; no measurable
wall-clock speedup at N=5/10/20. The aggregate-side cost is < 2% of
the profile at the scales declearn is realistically run at.
**Recommendation: do not merge.** The next worthwhile target for
masking-secagg scaling is websocket compression on the server's reply
path, not declearn's arithmetic.
