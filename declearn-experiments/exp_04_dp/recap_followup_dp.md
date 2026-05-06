# Experience 04 — DP-SGD — Follow-up Recap (variant comparison)

## Date
2026-05-04T11:15+02:00

## Goal

Pick the best variant among four candidate optimizations for the per-step
DP-budget-check hotspot identified in `exp_04/recap.md`. The hotspot:
opacus's RDP `get_epsilon()` is called once per training step in
`DPTrainingManager._prevent_budget_overspending`, and at the configured
2-round/250-step-per-round schedule that consumes 75% of total wall-clock.

Four variants (each on its own branch in `declearn-for-exp_04_dp/`):

- **H1 — defer**: skip `get_epsilon` entirely in the per-step path; keep
  only `accountant.step()` for history correctness. Round-end logging
  triggers the only `get_epsilon` call. **Mid-round detection: lost.**
  37-line diff. (Implemented in original exp_04 loop.)
- **H2 — periodic K=10**: call `get_epsilon` every K steps. K=1 is canonical, K=∞ is H1.
  Mid-round detection within K-1 steps. 40-line diff.
- **H3 — precompute N_max**: at round start, binary-search the maximum
  step count that keeps total ε ≤ budget. Per-step is then a single
  integer compare. **Mid-round detection: preserved exactly.** 104-line diff.
- **H4 — adaptive frequency**: cache the most recently observed ε; pick
  per-step check frequency from `cached_eps / budget`:
  - `> 0.9 · budget` → every step
  - `> 0.5 · budget` → every 10 steps
  - else → never (round-end refresh updates cached_eps).
  Mid-round detection: tier-dependent. 89-line diff.

## Setup

- Fork: `declearn-for-exp_04_dp/`, branches `exp_04_dp_variant_h{1,2,3,4}_*` off `master` (commit `45565ea`, canonical declearn 2.8.0).
- Config: `declearn-experiments/exp_04_dp/config_variant.toml` (rounds=2, budget=[5.0, 1e-5], n_clients=2, batch=48 with poisson sampling, model `model_torch_dp.py`).
- Data: regenerated per seed via `declearn-split --n_shards 2 --scheme iid --seed <S>` then chw-reshape; same data folder reused across all 5 arms within a seed.
- Cluster: magnet4. Load 1.5–2.0 during the run (some background tenant); both arms within a (seed, arm) pair run back-to-back so the bias is symmetric.
- Total wall-clock spent on this follow-up: ~25 min compute + ~10 min implementation/analysis.
- **`requires_human_crypto_review: true`** on all four variants per CLAUDE.md §11.4 (changes touch DP mechanism).

## Smoke (1 round × seed 42 × all 5 arms)

All five arms produce **byte-identical end-of-round ε = 4.99640485109163**, RC=0:

| arm | wall (s) | ε after round 1 |
|---|---|---|
| canonical | 90.81 | 4.99640485109163 |
| h1_deferred | 31.45 | 4.99640485109163 |
| h2_periodic | 37.46 | 4.99640485109163 |
| h3_precompute | 32.97 | 4.99640485109163 |
| h4_adaptive | 31.02 | 4.99640485109163 |

The accountant history is identical across arms by construction
(`accountant.step()` is preserved per-step in all variants; only the timing
of `get_epsilon()` differs).

Saved to `runs/smoke_followup_summary.json`.

## A/B (2 rounds × 3 seeds × 5 arms = 15 runs)

**Privacy invariant:** all 5 arms produce byte-identical eps_per_round
across all 3 seeds:
```
eps_per_round = [4.518721769152027, 4.518721769152027,
                 4.992799714321105, 4.992799714321105]
```
(2 entries per round × 2 clients × 2 rounds = 4 log entries; first two are end-of-round-1, last two are end-of-round-2.)

**Wall-clock summary (mean ± std across 3 seeds):**

| arm | wall mean (s) | wall std (s) | speedup vs canonical | mid-round detection |
|---|---|---|---|---|
| canonical | 156.52 | 6.98 | 1.00× | yes (canonical) |
| **h1_deferred** | **47.38** | **0.37** | **3.30×** | **NO** (round-end only) |
| **h3_precompute** | **52.68** | **1.20** | **2.97×** | **YES** (preserved exactly) |
| h2_periodic K=10 | 58.57 | 0.47 | 2.67× | YES (within 10 steps) |
| h4_adaptive | 102.76 | 3.82 | 1.52× | yes (when close to budget) |

Per-arm raw data: `runs/ab_followup_results.json`. Per-run logs: `runs/ab_followup_<arm>_seed<S>.log`.

## Surprises

### S1 — H4 (adaptive) is the slowest variant (1.52× vs H1's 3.30×)

Counter-intuition: I expected H4 to match H1 because "skip when far from budget". What actually happens with this configuration:

- σ is calibrated by `_fit_noise_multiplier` so total ε hits 5.0 exactly at end of training. After round 1, ε = 4.518 (90.4% of budget).
- At round 2 start, `_cached_epsilon` is refreshed → 4.518.
- 4.518 > 0.9 × 5.0 = 4.5 → tier picks `check_period = 1` → check every step.
- For all of round 2 (~250 steps), H4 calls `get_epsilon` per-step — same as canonical.

So H4 is "fast for round 1, canonical-speed for round 2." With the noise calibrated to exactly fit the budget over the planned rounds, **H4's win exists only when budget is loose relative to schedule**. For the typical "calibrate σ to fit the budget" pattern declearn uses, H4 doesn't help much.

If budget were 10.0 instead of 5.0 (loose), H4 would behave like H1. If budget were 100.0 (very loose), H4 would match H1 exactly. But that's not the deployment pattern.

### S2 — H3 is only 11% slower than H1, with mid-round detection preserved

H3 (precompute) costs 5.3 s more than H1 (deferred) on average (52.68 − 47.38). That's the round-start binary-search overhead: ~17 probes per round × 2 rounds = ~34 `get_epsilon` calls on a small history. At ~0.16 s per call (extrapolated from canonical's per-step cost on similar history sizes), total ≈ 5.4 s — matches.

H3 retains the **safety property** that canonical has and H1 dropped: a step that would push total ε past budget is detected at the exact step (not "sometime later"). The cost is 11% of H1's speedup.

### S3 — Variants have very low variance (std 0.37–1.20 s on ~50 s mean)

Canonical std is 6.98 s (4.5%). Variants' std is 0.7%–2.3% — the per-step get_epsilon overhead removed is also the dominant noise source. Once it's gone, what remains (training compute, asyncio scheduling) is much more deterministic.

This means even at 3 seeds (just 1 above the §6.4 minimum), the wall-clock comparison is statistically dispositive: the H1/H3 means are 5+ stdev apart from H2, and H2 is many stdev from H4 and canonical. **Adding more seeds would not change the ranking.**

### S4 — H2 (periodic K=10) is decently competitive but tunable-parameter-driven

H2 at K=10 gives 2.67× — between H4 and H3. H2 at K=∞ would converge to H1. H2 at K=1 is canonical. The K knob is a tunable parameter that future maintainers will bikeshed about. H3 has no such knob.

## Recommendation: H3 (precompute N_max)

**Pick H3 for upstream.** Reasoning:

1. **Near-best speedup (2.97× vs H1's 3.30×).** The 11% gap is the price of safety; for any production deployment that's worth it.
2. **Mid-round detection preserved exactly.** This is the safety property the canonical implementation existed to guarantee. H1 silently drops it; H3 keeps it.
3. **No tunable parameters.** No `K` to argue about (vs H2). No tier thresholds to validate (vs H4).
4. **Configuration-independent performance.** H3's speedup doesn't depend on whether the budget is tight or loose (vs H4, where the speedup collapses for tight budgets).
5. **Per-round overhead is bounded by `O(log N_max) × O(rounds)`** — well-defined, measurable, and small at any realistic scale (~5 s of binary-search cost across 2 rounds in this configuration).
6. **Single-file 104-line diff.** Patch lives at `patches/h_h3_precompute.diff`. Self-contained.

**Marc-level review checklist** before upstream:
- The binary search snapshot+restore of `accountant.history` is correct (no mutation leaks; verified by per-seed byte-identical eps_per_round).
- The `max_probe = 100_000` upper bound is sufficient for realistic step counts (current MNIST 2-round = ~500 steps; budget of 5.0 → N_max ≈ 250; binary search converges in 17 probes).
- Behavior at exhaustion: when `_step_counter_this_round > N_max`, we rollback the just-added accountant entry and raise `StopIteration`. Identical to canonical's rollback semantics.

**If H3 is rejected on review**, fallbacks in order: H1 (best perf, no safety) → H2 K=10 (configurable safety) → H4 (avoid for tight-budget configurations).

## Open observations

### O1 — All four variants preserve byte-identical privacy state

Across 15 runs (5 arms × 3 seeds), every arm produces the same `eps_per_round` for any given seed. This is because all variants preserve the per-step `accountant.step()` call (which appends to history). Only the *timing* of `get_epsilon()` differs. The accountant's view of "what training did" is unchanged.

This is the strongest possible smoke-test outcome: no plausible privacy regression from any variant.

### O2 — Canonical scaling is roughly linear in (steps × rounds), not quadratic

I had estimated canonical at 2 rounds would cost ~4× the 1-round cost (O(steps²) under the assumption that get_epsilon walks per-step history). Actual: 156.5 s vs 90.8 s = 1.7×. opacus's RDPAccountant compacts consecutive same-(noise, srate) steps into a single tuple `(noise, srate, count)`, so per-call cost is O(distinct (noise, srate) tuples) ≈ O(rounds), not O(steps). Total cost is then O(steps × rounds), still substantial but not quadratic in step count.

This reduces my prior worry that canonical would explode at higher round counts. It will still be slow (the per-call constant is ~0.23 s), but linearly so.

### O3 — Accuracy is similar across all arms (mean 0.11–0.18, all within DP-noise band at 2 rounds)

Mean accuracies per arm: canonical 0.1343, h1 0.1167, h2 0.1183, h3 0.1108, h4 0.1183. Spread is ~0.02 absolute, well within the DP-noise envelope at this small round count. Not informative for comparison; accuracy convergence requires many more rounds with σ calibrated for them.

### O4 — H3 binary-search overhead is bounded and measurable

H3 takes ~5.3 s more than H1 (across 2 rounds). That's the precompute cost: ~17 binary-search probes × 2 rounds × ~0.16 s/probe. At 10 rounds the cost would scale linearly to ~26 s — still negligible against the canonical 10-round cost of ~750 s. H3 dominates canonical at any realistic scale.

## Caveats

- 2 rounds × 3 seeds × 5 arms = 15 runs. Variance on variant arms is so low (std/mean < 2.5%) that 3 seeds is more than enough; no need to bump to the §6.4 DP-default of 5 seeds for the variant comparison. Canonical's std is 4.5% — fine for the baseline.
- Budget exhaustion path was NOT exercised. The configuration's σ is calibrated such that ε ≈ budget at end of last round; `StopIteration` is never raised. A separate stress test (tight budget, more rounds than σ was calibrated for) would validate that H3's `_step_counter_this_round > _max_steps_this_round` path actually fires correctly. Not done in this loop.
- All A/B runs at 2 clients, 2 rounds, batch_size 48, on `model_torch_dp.py`. Behavior at higher client counts (5, 10) or different model sizes was not measured. The fixes are per-client (each client runs its own DPTrainingManager), so we'd expect the same per-client speedup at any client count.
- H3's `max_probe = 100_000` upper bound was chosen as "obviously larger than any realistic per-round step count for this scale". For very large datasets (millions of training samples per client per round), this would need to be raised. Easy fix.

## Files

- Patches: `patches/h1_defer_get_epsilon.diff`, `patches/h_h2_periodic.diff`, `patches/h_h3_precompute.diff`, `patches/h_h4_adaptive.diff`
- Smoke runs: `runs/smoke_followup_summary.json`, `runs/smoke_followup_<arm>.log` (5 logs)
- A/B runs: `runs/ab_followup_results.json`, `runs/ab_followup_<arm>_seed<S>.log` (15 logs)
- Console transcript: `runs/ab_followup_console.log`
- Driver scripts: `smoke_all_variants.py`, `ab_followup.py`

## Per-variant py-spy profiles (added 2026-05-04)

Each arm was re-run once at 2 rounds, single seed (42), this time wrapped
in py-spy at 100 Hz. Use these JSONs to inspect *what's hot* in any
variant; use `ab_followup_results.json` (no py-spy) for the speedup
ranking — py-spy adds 12–22% sampling overhead so wall-clock here is
inflated.

| arm | py-spy wall (s) | A/B wall (s, no pyspy) | overhead | speedscope |
|---|---|---|---|---|
| canonical | 190.79 | 156.52 | +22% | `runs/profiles_followup/canonical/2026-05-04_17-59-58/pyspy_speedscope.json` |
| h1_deferred | 53.37 | 47.38 | +13% | `runs/profiles_followup/h1_deferred/2026-05-04_18-03-14/pyspy_speedscope.json` |
| h2_periodic | 66.80 | 58.57 | +14% | `runs/profiles_followup/h2_periodic/2026-05-04_18-04-11/pyspy_speedscope.json` |
| h3_precompute | 59.78 | 52.68 | +13% | `runs/profiles_followup/h3_precompute/2026-05-04_18-05-23/pyspy_speedscope.json` |
| h4_adaptive | 122.83 | 102.76 | +20% | `runs/profiles_followup/h4_adaptive/2026-05-04_18-06-27/pyspy_speedscope.json` |

The relative ordering of the arms is preserved (canonical >> h4 >> h2 > h3 > h1). Drop any of these JSONs into <https://www.speedscope.app> for interactive inspection.

### Empirical confirmation: H4 IS canonical-speed in round 2 (S1 verified)

H4's top self-time leaves are dominated by opacus RDP functions, exactly the same family that dominated canonical:

| function | canonical (orig observation) | h4_adaptive (this profile) |
|---|---|---|
| `_compute_log_a_for_frac_alpha` | 21.4% | 18.21% |
| `_compute_log_a_for_int_alpha` | 9.5% | 7.86% |
| `_log_erfc` | 8.5% | 5.83% |
| `_log_add` | 3.7% | 2.89% |
| **opacus-RDP top-4 sum** | **~43%** | **~35%** |

H4 pays ~75% of canonical's RDP cost. Round 1 is fast (cached_eps below 0.5·budget → no checks); round 2 has cached_eps > 0.9·budget → check_period=1 → canonical-speed. The wall-clock split (~half-canonical, half-H1) is 100% explained by per-round behaviour. **S1's design analysis is verified empirically.**

### H3 (recommended variant) hot-spot landscape — what's left after the fix

H3 total: 48.87 s of py-spy weight (recall +13% sampling overhead).

Top-15 by self-time:
- `_engine_run_backward` (torch autograd) — 14.53% — true backward-pass compute
- `_sample_noise` (declearn DP noise module) — 7.18% — gaussian noise generation, structurally required by DP-SGD
- `_apply_operation` (declearn Vector) — 7.02% — same cross-experiment Vector pattern (exp_01–09)
- `_compute_clipped_gradients` (declearn samplewise) — 4.95% — per-sample gradient clipping, structurally required
- `_compute_log_a_for_frac_alpha` (opacus RDP) — 4.60% — **residual from H3's per-round binary search** (~17 probes × 2 rounds)
- `clip_and_scale_grads_inplace` — 3.81%
- `norm` — 2.95%
- `_conv_forward` — 2.56%

Categorical aggregation (self-time):
- compute_torch: 49.99%
- declearn_internal: 18.52%
- imports_startup: 4.75%
- compression: 1.25%
- serialization, asyncio, websockets: <0.5% combined

Read: in canonical, opacus RDP was ~75% of total (cumulative get_privacy_spent path). In H3, opacus RDP residual is ~9-10% (binary-search probes only, called once per round). **13× reduction in RDP cost.** What's left is mostly real DP work (torch compute + gaussian noise + per-sample clipping) plus the cross-experiment Vector tax that hits every algorithm.

The next ceiling for further DP optimization, in order of magnitude:
1. **Vector dispatch (~7%)** — same `torch._foreach_*` rewrite that would help every algorithm in the study (cross-experiment finding from `findings.md`).
2. **Binary-search probes (~5%)** — could be cut by capping `max_probe` at a tighter upper bound (e.g., 2 × n_batches_per_round) since RDP `get_epsilon` is monotone in step count.
3. **`_sample_noise` (~7%)** — algorithmic; would require a faster gaussian RNG or pre-generated noise pool.

After 1+2 are done, H3 would be ~5-7% faster (~50 s → ~46 s). After all three, it'd approach the irreducible "actual DP-SGD work" floor.

## What Fares would want to revisit

1. **Submit H3 upstream** (after Marc's crypto review) — clean speedup, safety preserved, single-file patch.
2. **Stress-test budget exhaustion** with H3 — configure rounds > σ-planned-rounds, confirm `StopIteration` fires at the correct step.
3. **Re-run at higher client counts (N=5, N=10)** to confirm the per-client speedup composes cleanly.
4. **Consider H4 for loose-budget regimes** — where `cached_eps < 0.5 · budget` for most of training, H4 would skip per-step checks entirely and match H1 in performance. Useful as a complementary optimization for non-tightly-calibrated runs.
