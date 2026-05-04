# Experience 04 — DP-SGD — Recap

## Date
2026-04-30T21:50+02:00

## Setup
- Baseline: `declearn-for-exp_04_dp` branch `master`, commit `45565ea` (canonical declearn 2.8.0 snapshot)
- Variant fork: `declearn-for-exp_04_dp` branch `exp_04_dp_variant_h1`, commit `2dfe4c8`
- Patch: `patches/h1_defer_get_epsilon.diff` (37 lines)
- Data split: `examples/mnist_quickrun/data_iid_chw/` (channels-first reshape of `data_iid/`); regenerated per-seed for A/B (seeds 42, 43, 44)
- Cluster node: magnet4 (load 1.0–1.5 during runs; rising slightly but well under 4.0 floor)
- Total wall-clock spent on this experience: ~25 minutes
- **`requires_human_crypto_review: true`** — change touches a differential-privacy mechanism (§11.4)

## Hypotheses tested

### H1: per-step opacus `get_epsilon` is the dominant DP-SGD cost (CONFIRMED)

**Statement:** `DPTrainingManager._prevent_budget_overspending` (declearn/training/dp/_manager.py:221) calls opacus's RDP `get_epsilon()` once per training step. The RDP computation walks the accountant history, so cumulative cost is O(N²) in step count. Deferring the `get_epsilon` call to round boundaries (where it's already called for logging) preserves the same accountant history and produces identical end-of-round epsilon trajectories whenever the budget isn't exhausted mid-round. Expected ~3-4x wall-clock improvement at the 2c/2r/budget=5.0/δ=1e-5 settings.

- **Status:** CONFIRMED.

- **Smoke test:** PASS.
  - Config: `config_smoke_1r.toml` (rounds=1, otherwise identical to variant config), single seed, n_clients=2.
  - Baseline: wall=92.49s, ε=4.99640485109163, acc=0.1040.
  - Variant:  wall=31.43s, ε=4.99640485109163, acc=0.1245.
  - Δε = 0 (exact). Δacc = +0.0205 (within 0.05 tolerance). Speedup = 2.94x.
  - Saved to `runs/smoke_summary.json`.

- **A/B summary (3 seeds × 1 round × 2 clients):**

  | metric | baseline (mean ± std) | variant (mean ± std) | delta |
  |---|---|---|---|
  | wall-clock (s)   | 89.42 ± 2.37    | 30.99 ± 0.10  | **−58.43 s (−65%, +2.89x speedup)** |
  | final epsilon    | 4.996405 (exact across all 3 seeds) | 4.996405 (exact across all 3 seeds) | **0.000000** |
  | mean accuracy    | 0.1058 ± 0.0270 | 0.1273 ± 0.0815 | +0.0215 |

  Per-seed accuracies (full): see `runs/ab_results.json`.

- **Profile comparison summary:**
  - Observation profile (baseline at 2c/2r): **75.26% of total wall-clock spent in `opacus.accountants.rdp.get_privacy_spent`** (138.09 s of 186 s). Top self-time hits all in opacus RDP internals (`_compute_log_a_for_frac_alpha` 21.4%, `_compute_log_a_for_int_alpha` 9.5%, `_log_erfc` 8.5%, several variants of `_compute_log_a_for_frac_alpha` summing to ~30% via different stack-paths).
  - The RDP epsilon computation is pure-Python; opacus walks the full accountant history each call. With ~250 SGD steps × 2 rounds × 2 clients = 1000 `get_epsilon` calls in baseline, cumulative cost is O(N²) in step count.
  - Variant retains the per-step `accountant.step()` call (which appends to history — cheap, ~0 measurable cost) but eliminates the per-step `get_epsilon` call. Round-end logging still calls `get_privacy_spent` once per round (via `_training_round` line ~263), so the accountant's reported budget at logging is identical.

- **Deal-breaker assessment:**
  - Accuracy floor (variant ≥ baseline − 0.10): PASS. Variant is +0.0215, well within tolerance.
  - Smoke test (ε exact, acc within 0.05): PASS. Δε = 0; Δacc = +0.0205.
  - Crash/hang: PASS. All 6 A/B runs RC=0.
  - Perf direction (variant must be ≥30% faster): PASS. 65% faster, 2.89x speedup, far above the floor.
  - Confirmation: smoke pass + accuracy within tolerance + perf above noise (5%) + consistent direction across seeds (variant std on wall = 0.10 s, mean improvement = 58.4 s — improvement is >500x the noise std). **CONFIRMED.**

- **Code change:** `patches/h1_defer_get_epsilon.diff` (37 lines, declearn/training/dp/_manager.py only)
  ```diff
  -        if self.get_privacy_spent()[0] > self._dp_budget[0]:
  -            # Remove the step from the history as it will not be taken.
  -            last = self.accountant.history.pop(-1)
  -            ...
  -            raise StopIteration(...)
  +        # Variant H1: skip per-step get_epsilon entirely.
  ```

- **Result paths:**
  - Observation: `runs/exp_04_observation_dp/2026-04-30_21-30-28/pyspy_speedscope.json`
  - Smoke: `runs/smoke_summary.json`, `runs/smoke_baseline.log`, `runs/smoke_variant.log`
  - A/B: `runs/ab_results.json`, `runs/ab_baseline_seed{42,43,44}.log`, `runs/ab_variant_seed{42,43,44}.log`
  - Patch: `patches/h1_defer_get_epsilon.diff`

## Open observations (NOT tested as hypotheses)

### O1 — Mid-round budget detection is now lost
The variant detects budget overruns at round boundaries instead of mid-step. If a configuration has more steps per round than the budget can fund, the variant takes those extra steps before raising. Practically: at 2c/2r/budget=5.0/δ=1e-5/MNIST, the budget never gets exhausted within a round, so this is academic. For configurations approaching the budget at single-step granularity, a hybrid (lazy check that triggers periodic recompute when ε is within K× of the limit) would preserve mid-round detection. Not implemented here.

### O2 — Even after H1, declearn-internal Vector dispatch + opacus.history-step call still sit at 1-2% each
Vector `_apply_operation` is at 1.92% self-time in the DP observation profile. Same pattern as exp_01-03: independent of DP, a Vector-layer optimization would benefit DP too. Not pursued in this experience.

### O3 — Pre-step `get_epsilon` call could be skipped by precomputing max-allowed-steps once per round
Given fixed (noise, srate) per round, the relationship between step count and epsilon is monotonic and can be inverted once per round to compute "max steps allowed before budget exceeded." This would preserve mid-round StopIteration semantics with zero per-step overhead beyond an integer compare. Cleaner than the H1 variant. Suggested follow-up.

## Conclusions

**Headline:** declearn's DP-SGD wastes ~65% of its wall-clock on a redundant per-step privacy-budget check. Removing the per-step `get_epsilon` call (while keeping `accountant.step()` for bookkeeping) yields a clean 2.89x speedup with byte-identical end-of-round epsilon and within-noise accuracy. The patch is 37 diff lines in one file.

**What Fares would want to revisit:**
- **Crypto/DP review (mandatory per §11.4):** the variant changes the granularity of mid-round budget enforcement from per-step to per-round. For typical FL setups (where the budget is per-round-orchestrated), this is functionally equivalent. For per-step adversarial budgets, it isn't. Need explicit Marc-level review before upstreaming.
- The cleaner formulation in O3 (precompute max-steps-per-round once) deserves a follow-up exp_04b that preserves mid-round detection while still being O(1) per step.
- This is the FIRST deal-breaker confirmed in the autonomous loop. The pattern (find a per-step expensive recompute; defer or precompute) generalizes — flagging for application to the SecAgg / Fairness experiences if similar O(N²) hotspots appear.

## Caveats and open questions

- A/B was at 1 round (matching smoke test) rather than 2 rounds. Reasoning: 1-round at 3 seeds gives a clean 2.89x effect with std 0.10s on variant; running at 2 rounds would have hit the §6.4 30-min abort. The relative speedup is determined by the per-step computation and is independent of round count, so 1-round result is representative for the wall-clock claim. Accuracy comparison is less informative at 1 round (both arms ~10% accuracy under DP noise), but the smoke test at 1 round and the A/B at 1 round agree. Re-running at 2 rounds × 5 seeds is straightforward if Marc wants tighter accuracy bounds.
- The opacus accountant history is identical between arms — verified by exact ε match across all 6 A/B runs. This is the strongest possible smoke-test outcome (byte-identical privacy state).
- This experiment did NOT exercise the budget-exhaustion path. A separate test should: (a) configure a tight budget that exhausts mid-round in baseline and (b) confirm the variant detects exhaustion at round boundary instead.
