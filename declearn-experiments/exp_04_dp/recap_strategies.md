# Experience 04 — DP-SGD budget-check strategies — Recap

## What this experiment was

Once we identified (in `recap.md`) that `_prevent_budget_overspending` calls
opacus's `get_epsilon()` on every training step and that this consumes
75 % of total wall-clock, the question became: **what's the right way to
fix it?**

There are three obvious-shaped fixes — periodic, precompute, adaptive —
plus a fourth "deferred" one that we already had from the first H1 run.
Each makes a different trade-off between wall-clock cost, mid-round
detection of budget overspend, and implementation complexity. This
follow-up tested all four side-by-side to pick the one worth upstreaming.

## The four strategies, plain English

**H1 — defer.** Skip `get_epsilon()` entirely in the per-step path. The
accountant's `step()` call (which appends to history) still runs every
step so the privacy state stays correct. The only `get_epsilon()` call
is the one declearn already made at the end of every round for logging
— that becomes the *de facto* budget check too. Trade-off: if a
configuration somehow over-trains beyond what σ was calibrated for, we
detect it one round late, not one step late.

**H2 — periodic K=10.** Same as canonical, but only call `get_epsilon()`
every 10th step instead of every step. Tunable: K=1 is canonical, K=∞
is H1. Trade-off: budget overspend is detected within K-1 steps of the
actual exhaustion. Simple to implement, but introduces a parameter
people will argue about.

**H3 — precompute N_max once per round.** At round start, ask: "given
the current ε, the noise/sampling parameters, and the privacy budget,
what's the largest number of steps I can take in this round before
exceeding the budget?" Call that number `N_max`. Then per-step is just
`if step_counter > N_max: stop`. The precompute uses binary search over
the RDP accountant — ~17 probes per round, each cheap because opacus
stores history compactly. Trade-off: 5 s of precompute work per round
in exchange for keeping mid-round detection *exactly* as canonical had
it.

**H4 — adaptive frequency.** Cache the most recently observed ε. Pick
the per-step check frequency from how close cached_eps is to the budget:
- `> 0.9 · budget` → check every step
- `> 0.5 · budget` → check every 10 steps
- otherwise → never check (round-end log refreshes cached_eps)

Idea: when you're at €0.50 of a €5 budget, no single step can blow it,
so don't bother. When you're at €4.80, check every step.

## How we tested them

- Each variant was implemented on its own branch in
  `declearn-for-exp_04_dp/` off the canonical baseline (`master`).
- **Smoke**: each variant ran 1 round, single seed (42), and we confirmed
  the end-of-round ε was byte-identical across all 5 arms (canonical
  4.99640485109163, all variants 4.99640485109163). This proved the
  variants don't change the privacy semantics — only the timing of when
  the budget check fires differs.
- **A/B**: 5 arms × 3 seeds × 2 rounds = 15 runs, no py-spy (so the
  wall-clock numbers are honest production-like). Cross-arm
  ε-trajectory equivalence was checked per seed; all matched.
- **Inspection**: each variant was re-run once with py-spy at 100 Hz so
  the speedscope JSONs could be opened in
  <https://www.speedscope.app> for "what's hot now" investigation.

## What we found

### Wall-clock (3-seed mean, no py-spy)

| arm | wall (s) | std | speedup vs canonical | mid-round detection |
|---|---|---|---|---|
| canonical | 156.52 | 6.98 | 1.00× | yes |
| h1_deferred | 47.38 | 0.37 | 3.30× | **lost** |
| **h3_precompute** | **52.68** | **1.20** | **2.97×** | **preserved** |
| h2_periodic K=10 | 58.57 | 0.47 | 2.67× | within 10 steps |
| h4_adaptive | 102.76 | 3.82 | 1.52× | yes (when close) |

### The expected shape held for H1, H2, H3

H1 is fastest because it does the least work per step. H3 trails H1 by
~5 s (= the round-start binary search). H2 trails H3 by ~6 s (= the
10 % of steps that still call `get_epsilon`). Variance on all three is
tiny (std ≤ 1.2 s on a ~50 s run, < 2.5 %).

### H4 was the surprise

H4 was supposed to give H1-class performance most of the time, falling
back to canonical-speed only near the budget. In practice it was 1.52×
— barely better than canonical for round 2.

The reason is structural: declearn's `_fit_noise_multiplier` calibrates
σ so that ε hits the budget *exactly* at end-of-training. After round 1,
ε ≈ 4.52 (out of budget 5.0) — that's 90.4 % of budget, above H4's
0.9-budget threshold. So for all of round 2, H4's tier picks
`check_period = 1` and degenerates to canonical.

The py-spy profile confirms this empirically: the same opacus RDP
functions that dominated canonical (`_compute_log_a_for_frac_alpha` at
21.4 %, `_compute_log_a_for_int_alpha` at 9.5 %, `_log_erfc` at 8.5 %)
reappear in H4's profile at 18.21 %, 7.86 %, 5.83 % — about 75 % of
canonical's RDP cost. H4 is "round 1 fast, round 2 canonical."

**Adaptive is only useful when the budget is loose.** The typical
declearn deployment pattern (calibrate σ to fit the planned schedule)
makes the budget tight by construction. H4's design assumes the
opposite assumption.

### What's left after the fix in H3

We profiled H3 with py-spy too. Top hotspots in the recommended
variant, in order of self-time:

| frame | self % | what it is |
|---|---|---|
| `_engine_run_backward` | 14.5 % | torch autograd — real backward pass |
| `_sample_noise` | 7.2 % | DP gaussian noise — algorithmically required |
| `_apply_operation` (Vector) | 7.0 % | the cross-experiment Vector dispatch tax |
| `_compute_clipped_gradients` | 5.0 % | per-sample gradient clipping |
| `_compute_log_a_for_frac_alpha` | 4.6 % | residual from H3's per-round binary search |
| `clip_and_scale_grads_inplace` | 3.8 % | grad clipping helper |

Compared to canonical (where opacus RDP totalled ~75 %), H3's residual
RDP cost is ~9-10 % (binary-search probes only, called once per round).
**13× reduction in RDP cost.** What's left is mostly real DP work —
torch compute + gaussian noise + per-sample clipping — plus the
cross-experiment Vector tax.

## Pick

**H3 — precompute N_max once per round.**

Why:

1. **97 % of the maximum achievable speedup.** The 11 % gap to H1 is
   the price of safety; for a privacy mechanism this is a good trade.
2. **Mid-round detection preserved exactly.** This is the canonical
   safety property; H1 silently drops it.
3. **No tunable parameter.** No K to bikeshed about (vs H2). No tier
   thresholds to validate (vs H4).
4. **Configuration-independent.** Works for tight or loose budgets —
   unlike H4, which collapses for the typical declearn deployment.
5. **Bounded per-round overhead.** ~17 binary-search probes × cheap
   `get_epsilon` calls (history is compact in opacus) ≈ ~5 s in the
   tested configuration. Scales linearly in rounds, not in steps.
6. **Single-file, 104-line patch.** Self-contained.

Fallbacks if H3 is rejected on review: H1 (best perf, no safety) → H2
K=10 (configurable safety) → avoid H4 for tight-budget configurations.

## What's not in the data yet

- **Budget exhaustion path.** σ was calibrated to fit ε ≤ 5.0 over 2
  rounds, so the `StopIteration` rollback path was never exercised. A
  separate stress test (rounds > σ-planned) is needed to confirm H3
  actually catches over-budget at the correct step. Easy to add.
- **Higher client counts.** All A/B runs at 2 clients. The fix is
  per-client (each client's `DPTrainingManager` is independent), so the
  speedup should compose, but this hasn't been measured at N=5 or
  N=10.
- **Larger models / longer training.** RDP per-step cost scales
  (slowly) with the number of distinct (noise, srate) tuples in
  history. For >>2 rounds with the same σ, history stays at length 1,
  so canonical's per-step cost is roughly constant. Variants are
  insensitive to this.

## Where the data lives

| if you want… | open… |
|---|---|
| The wall-clock ranking | `runs/ab_followup_results.json`, table in `recap_followup_dp.md` |
| What's hot in canonical (the problem) | `runs/exp_04_observation_dp/.../pyspy_speedscope.json` |
| What's hot in H3 (the recommended fix) | `runs/profiles_followup/h3_precompute/.../pyspy_speedscope.json` |
| Empirical proof of the H4 result | `runs/profiles_followup/h4_adaptive/.../pyspy_speedscope.json` |
| The H3 patch | `patches/h_h3_precompute.diff` |
| The 15 raw A/B logs | `runs/ab_followup_<arm>_seed<S>.log` |

The full follow-up methodology and detailed numbers are in
`recap_followup_dp.md`. This file is the narrative summary; that one is
the data sheet.

## Lessons (process, not declearn)

1. **Don't trust an optimization's design without measuring it.** I
   predicted H4 would be H1-class. It wasn't. The py-spy profile would
   have shown me why in five minutes.
2. **The opposite of "always check" is rarely "never check" — usually
   it's "check at the right time."** H3's "compute the answer once, use
   it many times" pattern is more durable than H1's "skip the question
   entirely" pattern.
3. **Variants are easier to reason about when only one thing changes
   between them.** All four variants here keep `accountant.step()` per
   step (so the accountant history is identical to canonical). Only
   the `get_epsilon()` timing differs. That made the privacy
   equivalence verifiable at smoke time and made the comparison about
   wall-clock alone.
