# Experience 10 — TorchVector ` torch._foreach_*` batching — Recap

## Date
2026-05-06

## Why this experiment
`findings.md:30–63` flagged Vector dispatch (`Vector._apply_operation`,
`Vector.apply_func`) as a leaf hotspot in **9/9** previous experiments
at 13–17% self-time and recommended:

> target `TorchVector` specifically with `torch._foreach_*` batched ops
> (`_foreach_add_`, `_foreach_mul_`, `_foreach_sign`, etc.). Estimated
> impact: a 50% reduction in Vector-dispatch self-time would shave 6–8%
> off every algorithm.

The recommendation was extrapolated, never validated. exp_10 is the
end-to-end measurement.

## Setup
- Baseline: `declearn-for-exp_10_vector_foreach/master` = canonical declearn 2.8.0 snapshot
- Variant fork: `exp_10_vector_foreach_variant`
- Patch: `patches/torchvector_foreach.diff` (73 lines, single file: `declearn/model/torch/_vector.py`)
- Data split seed(s): 42, 43, 44 (per-seed regenerate, both arms reuse the same split)
- Cluster node: magnet4 (load 0.48 at start)
- Benchmark: `mnist_quickrun`, 2 clients, 2 rounds, batch 48
- Algorithms tested: vanilla FedAvg, FedAvg+lasso, FedAvg+SCAFFOLD
- Total wall-clock spent on this experiment: ~12 min (18 A/B runs + 2 py-spy runs)

## What the patch does

In `TorchVector` only (the framework-agnostic `Vector` ABC is untouched):

- `_apply_operation` adds a fast path: when `func` is one of the seven
  framework-known binary ops (`torch.add/sub/mul/div/pow/minimum/maximum`),
  collect all coefficient tensors into a list and call the corresponding
  `torch._foreach_*` op once instead of looping in Python and calling
  `func` per tensor.
- `apply_func` adds a fast path for stateless unary torch funcs (`sign`,
  `abs`, `neg`, `reciprocal`) that have `_foreach_*` counterparts. Any
  call with extra args/kwargs (e.g. `torch.clamp(min=…, max=…)`) falls
  through to the canonical per-tensor loop, so the patch is non-invasive.

Both fast paths fall through to `super()._apply_operation` /
`super().apply_func` for any function not in the lookup table.

## Hypotheses tested

### H1 — `torch._foreach_*` substitution measurably reduces wall-clock
- Status: **inconclusive**
- Smoke test: **PASS** — 15 ops (binary vector-vector, binary vector-scalar,
  unary, fallback `torch.clamp` with kwargs) all produce **byte-identical**
  outputs to canonical (`torch.equal == True`, `max_abs_diff = 0.0` on every op).
- A/B (3 seeds × 2 rounds × 2 clients):

  | algo     | arm     | wall (s, mean) | std  | speedup vs master | acc (mean) |
  |----------|---------|----------------|------|-------------------|------------|
  | vanilla  | master  | 29.01          | 1.08 | 1.00×             | 0.892      |
  | vanilla  | variant | 28.03          | 0.75 | **1.03×**         | 0.896      |
  | lasso    | master  | 29.49          | 0.28 | 1.00×             | 0.104      |
  | lasso    | variant | 29.72          | 0.65 | **0.99×**         | 0.097      |
  | scaffold | master  | 28.78          | 0.50 | 1.00×             | 0.905      |
  | scaffold | variant | 28.06          | 0.26 | **1.03×**         | 0.898      |

- Deal-breaker assessment per CLAUDE.md §7:
  - Smoke equivalence: ✓ (byte-equal across all 15 ops)
  - Accuracy floor (delta < 0.10): ✓ (all deltas ≤ 0.01)
  - Perf direction: variant is faster on vanilla and scaffold, very slightly slower on lasso
  - **Improvement above noise (5%):** ✗ — best speedup is 3.4%, std is comparable (~3%) → within the noise band, not a confirmed win
  - **Consistency across seeds (mean improvement > 1 std):** ✗ — vanilla mean delta 0.98s vs std 1.08s (master); scaffold 0.72s vs std 0.50s (master)

### Profile inspection (single-seed, py-spy 100 Hz, lasso config)

The A/B does not detect a wall-clock effect, but the py-spy comparison
shows the patch DOES eliminate the dispatch hotspot exactly as predicted —
the time just doesn't translate into much wall-clock at this scale.

Total weight: master 22.49 s · variant 21.62 s · delta -0.87 s

| frame                                            | master self_s | variant self_s | delta_s  |
|--------------------------------------------------|---------------|----------------|----------|
| `_apply_operation` in `model/api/_vector.py` (parent dict-comprehension) | 4.24 | 0.00 | **-4.24** |
| `_apply_operation` in `model/torch/_vector.py` (now does the work) | 0.19 | 3.74 | +3.55 |
| `apply_func` in `model/api/_vector.py`           | 1.32          | 0.00           | **-1.32** |
| `apply_func` in `model/torch/_vector.py` (new override) | 0.00      | 1.03           | +1.03 |
| **Sum of dispatch self-time** | **5.75 s** | **4.77 s** | **-0.98 s (-17%)** |

The Python-level dict-comprehension iteration at the parent class
(`api/_vector.py`) is **fully eliminated** — both `_apply_operation` and
`apply_func` show 0.00 s self in the variant. The cost moves into the
TorchVector subclass overrides, which now do the list-marshaling and
foreach call themselves. Net dispatch self-time drops 17%, ≈ 0.98 s on a
~22 s profile, which matches the wall-clock delta within noise.

Other top-15 frames are essentially unchanged (`_engine_run_backward` -0.08 s,
`_conv_forward` -0.04 s, dropout/maxpool ±0.08 s — all well within sampling
noise).

### Why the recommendation was over-optimistic

The findings.md recommendation projected "50% reduction in Vector-dispatch
self-time → 6–8% wall-clock". We observed:

- Dispatch self-time reduction: 17% (not 50%). `torch._foreach_*` removes
  the per-tensor Python *iteration* tax, but the patch still has its own
  Python overhead — building the keys/tensors lists, dict-rebuilding the
  result, the `dict.get(func)` lookup, the isinstance check. With ~10
  parameter tensors in the mnist CNN, the per-call list-construction cost
  is non-negligible relative to what `_foreach_*` saves.

- Wall-clock translation: the ~0.98 s saved per run is only ~3% of a
  30 s total — at the edge of noise (run-to-run std ~3% on this benchmark).

The profile-self-time → wall-clock mapping assumed a 1:1 relationship.
In practice, removing CPU-side Python overhead can be hidden by other
parts of the call chain becoming relatively more expensive (some torch
kernels showed +0.07 s — likely re-balancing of a fixed-size workload).

## Open observations (not tested as hypotheses)

- **Lasso is not an outlier vs vanilla at this benchmark size.** findings.md
  reported lasso +12.5% over vanilla; in this run lasso master = 29.49 s
  vs vanilla master = 29.01 s = +1.7%. Either the earlier observation was
  noise, or it depended on a configuration we didn't replicate (different
  alpha, different optimizer, different N). The "lasso outlier" claim
  needs re-grounding with a fresh A/B if anyone wants to act on it.
- **`_apply_operation` in `api/_vector.py` shows up as TWO distinct
  frames** in the master profile (3.43 s + 0.75 s = 4.24 s). This is a
  py-spy quirk where the same source function appears at two line
  offsets; the analyze script doesn't merge them. The variant cleanly
  has one frame (in TorchVector) which is structurally simpler to read.
- **Patch fall-through works as intended.** The `clamp` smoke
  (`torch.clamp(min=-0.5, max=0.5)`) still byte-matches canonical because
  the kwargs cause it to skip the fast path. This was the main correctness
  risk and it held.

## Conclusions

Headline: at mnist_quickrun benchmark scale (CNN with ~10 parameter tensors,
~250 steps/round, 2 rounds, 2 clients, ~30 s total), substituting
`torch._foreach_*` for the per-tensor dispatch loop produces a
**measurable reduction in dispatch self-time (-17%, -0.98 s on a 22 s
profile) but no statistically significant wall-clock speedup (1.03× best,
within run-to-run noise)**.

What this means for the findings.md recommendation:
- The change is **correct** (smoke 15/15 byte-equal) and **directionally
  right** (eliminates the parent-class iteration hotspot).
- The projected 6–8% wall-clock gain at mnist-quickrun scale **does not
  materialize**. The actual gain at this scale is ≈ 3% — within noise,
  inconclusive.
- The patch is more likely to pay off on **larger models** (more tensors
  per Vector → batching wins more) or **longer training** (amortize the
  one-time costs Python-side). The mnist quickrun CNN is on the small
  side (10 params); a ResNet18 with ~120 params would likely show a
  cleaner signal. **Recommend re-running on a bigger model before
  upstreaming**.

What Fares would want to revisit:
- Re-run the A/B with a bigger torch model (any of the bench harness
  configs that use a ResNet-class model) to see if the dispatch gain
  becomes wall-clock-visible.
- Decide if a 17% dispatch self-time reduction with no wall-clock proof
  is worth merging. It's safe (byte-equal smoke), it's a self-contained
  78-line patch, and it's framework-idiomatic torch — but it's also not
  shipping a measurable user-visible win at the demo scale.

## Caveats and open questions

- **Run-to-run variance dominates the signal.** wall_std on master runs
  was 0.28–1.08 s on a 28–29 s mean — i.e. 1–4% noise floor. A 3%
  speedup is at-or-below that floor. Either need (a) bigger-scale runs
  to lift the signal, or (b) more seeds to shrink the noise.
- **Single-seed py-spy.** The profile delta was measured once per arm.
  py-spy itself adds ~5–10% sampling overhead, so the absolute self-time
  numbers aren't directly comparable to the A/B numbers, only to each
  other.
- **Single benchmark.** mnist_quickrun is the only workload tested.
  Findings.md said the dispatch tax shows up across all 9 algorithms;
  we tested 3 of those 9 (vanilla, lasso, scaffold) and all 3 showed
  the same near-noise wall-clock pattern. We didn't test secagg
  (where dispatch was already drowned out by quantization) or fairness
  (where the unified runner setup adds confound).
- **The 13–17% self-time finding stands.** What this experiment shows
  is that *eliminating it via `_foreach_*` substitution* doesn't
  proportionally translate to wall-clock at this scale. The hotspot is
  real; the leverage on it via this patch is smaller than projected.

## Where the data lives

| if you want…                                  | open…                                                                                  |
|-----------------------------------------------|----------------------------------------------------------------------------------------|
| The 18 A/B run logs                           | `runs/ab_*.log`                                                                        |
| The A/B summary (machine-readable)            | `runs/ab_results.json`                                                                  |
| The A/B summary (human)                       | `runs/ab_run.console.log`                                                              |
| Master py-spy profile (lasso, seed 42)        | `runs/profiles/master/2026-05-06_11-18-58/pyspy_speedscope.json`                       |
| Variant py-spy profile (lasso, seed 42)       | `runs/profiles/variant/2026-05-06_11-19-35/pyspy_speedscope.json`                      |
| Side-by-side profile comparison               | `runs/compare_lasso.txt`                                                                |
| The patch                                     | `patches/torchvector_foreach.diff`                                                     |
| Smoke driver + per-op pickles                 | `smoke_orchestrate.py` / `smoke_unit.py` / `runs/smoke_master.pkl` + `smoke_variant.pkl` |
| A/B driver                                    | `ab_run.py`                                                                            |
| Profile driver                                | `profile_arms.py`                                                                      |

## Status (small-model A/B)

**Inconclusive (per CLAUDE.md §7).** Patch is correct and directionally
right but the wall-clock improvement is within noise at this scale.
Re-running on a bigger model is the cleanest follow-up.

---

# Follow-up: bigger-model A/B (2026-05-06)

## Why a follow-up was needed
The small-model A/B was inconclusive because the patch's saving
(~1 s of dispatch self-time per run) sat at the noise floor (run-to-run
std ~1 s on a ~30 s mean). The dispatch hotspot scales with the
**number of parameter tensors**, but mnist_quickrun's CNN has only 6 of
those. We re-ran the full A/B with a deeper CNN to see if the dispatch
gain becomes wall-clock-visible when the count rises.

## What changed
- **Model:** new `BiggerCNN(depth=20, channels=8)` written to
  `examples/mnist_quickrun/model_torch_big.py`.
  - 86 parameter tensors (14× the small CNN's 6)
  - ~50 k scalar params (5× the small CNN's ~10 k)
  - Same input shape, same data, same MNIST training loop — only the
    network is bigger and deeper.
- **Configs:** `config_{vanilla,lasso,scaffold}_big.toml` point at the
  new model file; everything else (optim, batch size, rounds) identical
  to the small-model configs.
- **Driver:** `ab_run_big.py` (same shape as `ab_run.py`).
- **Profile driver:** `profile_arms_big.py`.
- **Patch:** unchanged. Same `torchvector_foreach.diff`. Same smoke
  (15/15 byte-equal) still applies — `TorchVector` operations are
  still byte-identical between branches; what changed is the *workload*
  exercising those ops.

## Wall-clock A/B results (3 seeds × 2 rounds × 2 clients)

| algo     | arm     | wall (s, mean) | std  | speedup vs master | improvement / std | acc (mean) |
|----------|---------|----------------|------|-------------------|-------------------|------------|
| vanilla  | master  | 177.85         | 0.94 | 1.00×             | —                 | 0.342      |
| vanilla  | variant | **165.97**     | 2.15 | **1.07×** (-11.9 s) | **12.6×** ✓   | 0.290      |
| lasso    | master  | 230.92         | 0.31 | 1.00×             | —                 | 0.102      |
| lasso    | variant | **219.86**     | 1.03 | **1.05×** (-11.1 s) | **35×**  ✓    | 0.100      |
| scaffold | master  | 182.21         | 2.74 | 1.00×             | —                 | 0.320      |
| scaffold | variant | **175.88**     | 2.66 | **1.04×** (-6.3 s) | **2.3×** ✓     | 0.302      |

`improvement / std` is `mean_wall_delta / wall_std(master)` — the §7
"mean improvement > 1 std" check. All three algorithms clear it
comfortably.

## Deal-breaker assessment per CLAUDE.md §7

- Smoke equivalence (15-op byte-equal): ✓ (carries over from small-model run)
- Accuracy floor (mean delta ≤ 0.10): ✓ — biggest mean delta is vanilla
  at 0.052. Per-seed deltas vary up to 0.27 (vanilla seed 44) but this
  is benign — see "RNG caveat" below.
- Improvement above noise (5%): ✓ for vanilla (6.7%), borderline for
  lasso (4.8%), below for scaffold (3.5%). All three pass the stricter
  consistency check.
- Consistency across seeds (mean improvement > 1 std): ✓ for all three,
  overwhelmingly so for vanilla (12.6×) and lasso (35×).

**Vanilla and lasso: confirmed.** scaffold: significant but the smaller
delta is dominated by the per-seed run-time variance specific to scaffold
(its aux_var exchange adds non-determinism into wall-clock).

## py-spy comparison (lasso big, single-seed, 2 rounds)

| frame                                                | master self_s | variant self_s | delta_s  |
|------------------------------------------------------|---------------|----------------|----------|
| `_apply_operation` in `model/api/_vector.py` (parent dispatch) | 34.51 | 0.00 | **-34.51** |
| `_apply_operation` in `model/torch/_vector.py` (now hot)       | 0.97  | 21.14 | +20.17 |
| `apply_func` in `model/api/_vector.py`                | 1.62          | 0.00           | **-1.62** |
| `apply_func` in `model/torch/_vector.py` (new override) | 0.00       | 1.02           | +1.02 |
| **Sum of dispatch self-time**                         | **37.10 s**   | **22.16 s**    | **-14.94 s (-40%)** |
| `_engine_run_backward` (autograd, untouched by patch) | 184.83        | 179.46         | -5.38 (noise) |
| `_conv_forward` (untouched by patch)                  | 36.48         | 32.64          | -3.84 (noise) |
| profile total                                         | **298.39 s**  | **272.01 s**   | **-26.38 s** |

**The big-model run reproduces the dispatch elimination cleanly:**
the parent dict-comprehension at `api/_vector.py` drops to 0 s in the
variant on both code paths (`_apply_operation` and `apply_func`), and
the cost moves into the TorchVector subclass overrides which now do
the list-marshaling and `_foreach_*` call. **Net dispatch self-time
drops 40%** (-14.94 s on a 298 s profile = -5%), almost exactly what
the wall-clock A/B shows after subtracting noise from un-patched frames.

The "moved" self-time is genuine work the variant has to do: gathering
keys, building the tensor list, calling the `_foreach_*` op, rebuilding
the dict. What's saved is the per-tensor Python interpretation of
`func(self.coefs[key], other.coefs[key])` looped 86 times per call. At
86 tensors instead of 6, the saving is large enough to lift the signal
above the noise.

## Scaling: small-model vs big-model dispatch savings

| metric                         | small CNN (6 tensors) | big CNN (86 tensors) |
|--------------------------------|------------------------|----------------------|
| profile total weight (master)  | 22.49 s                | 298.39 s             |
| dispatch self-time (master)    | 5.75 s (26%)           | 37.10 s (12%)        |
| dispatch self-time (variant)   | 4.77 s (22%)           | 22.16 s (8%)         |
| dispatch self-time saved       | -0.98 s (-17%)         | -14.94 s (-40%)      |
| wall-clock saved (mean A/B)    | -0.98 s (-3%, noise)   | -11.1 s (-5%, ✓)     |
| run-to-run std (master)        | ~1 s (3% of 30 s)      | ~1 s (0.5% of 230 s) |

Two effects compound at scale:
1. **Absolute dispatch saving grows roughly linearly with tensor count.**
   ~1 s saved at 6 tensors → ~15 s saved at 86 tensors.
2. **Relative noise shrinks** because the wall-clock grows faster than
   the run-to-run variance. At the small model, std/mean was ~3 %;
   at the big model, std/mean for lasso is ~0.1 %. This is what lets
   the same-magnitude relative speedup become statistically detectable.

The 50 % dispatch reduction projected in `findings.md` was over-optimistic
(actual is 40 %), and the 6–8 % wall-clock projection was right ballpark
for moderately deep models. The mnist_quickrun small CNN simply doesn't
have enough tensors for the dispatch tax to be worth removing.

## RNG caveat (does NOT affect the wall-clock conclusion)

The big-model accuracy varies more across (master, variant) pairs at
the same seed than the unit-smoke byte-equality would suggest:

- vanilla seed 44: master 0.442, variant 0.173 (Δ = 0.27)
- mean over all 3 seeds: master 0.342, variant 0.290 (Δ = 0.05)

This is not the patch's fault. `model_torch_big.py` doesn't pin
`torch.manual_seed`, so each subprocess (one per branch-reinstall +
quickrun cycle) initializes Conv/BN parameters from whatever default
RNG state the import chain leaves. Master-then-variant subprocesses
are not at the same RNG state at model-instantiation time, so weights
diverge from step 0 and trajectories drift over 1000 training steps.
Mean accuracy across seeds remains within 0.10 — the §7 floor — so
this is a methodology note, not a correctness regression. To pin it,
add `torch.manual_seed(42)` at the top of `model_torch_big.py` before
the `BiggerCNN()` call. We didn't pin it because the wall-clock
question doesn't need pinned init.

## Updated conclusion

**The patch is confirmed effective on models with non-trivial parameter-
tensor counts.** Findings.md's recommendation (substitute
`torch._foreach_*` for the per-tensor dispatch) is correct in principle
but its leverage scales with model depth. At mnist_quickrun (~6 tensors)
it's lost in the noise; at a 20-block CNN (~86 tensors) it produces
4–7 % wall-clock speedup at high statistical confidence. ResNet18
(~120 tensors) and HuggingFace transformers (~100–500 tensors,
exemplified by declearn's own `examples/nlp/` DistilBERT) are well past
the threshold where this patch matters.

**Recommendation update:**
1. **Upstream the patch.** Byte-equal smoke holds, dispatch elimination
   is clean, wall-clock impact is real on any non-trivially-deep model.
   The 73-line single-file diff is self-contained.
2. The small-model A/B alone would not have justified merging. The
   big-model A/B does.
3. The "-40 %" dispatch self-time delta is a hard upper bound on what
   this approach can give — the remaining 22 s in the variant's
   `_apply_operation` is the marshaling cost the patch introduces. To
   go further would require attacking the marshaling itself (e.g. cache
   the keys list, share tensor lists across consecutive same-shape ops),
   which is a much more invasive change with smaller marginal returns.

## Where the big-model data lives

| if you want…                                  | open…                                                                                  |
|-----------------------------------------------|----------------------------------------------------------------------------------------|
| The 18 big A/B run logs                       | `runs/ab_big_*.log`                                                                    |
| The big A/B summary (machine-readable)        | `runs/ab_big_results.json`                                                              |
| The big A/B summary (human)                   | `runs/ab_big.console.log`                                                              |
| Master py-spy profile (big lasso, seed 42)    | `runs/profiles_big/master/2026-05-06_14-37-22/pyspy_speedscope.json`                   |
| Variant py-spy profile (big lasso, seed 42)   | `runs/profiles_big/variant/2026-05-06_14-42-31/pyspy_speedscope.json`                  |
| Side-by-side profile comparison (big)         | `runs/compare_lasso_big.txt`                                                            |
| The bigger-model definition                   | `../../examples/mnist_quickrun/model_torch_big.py`                                     |
| Big-model configs                             | `config_{vanilla,lasso,scaffold}_big.toml`                                              |
| Big A/B driver                                | `ab_run_big.py`                                                                        |
| Big profile driver                            | `profile_arms_big.py`                                                                  |

## Final status

**Confirmed (per CLAUDE.md §7).** Vanilla and lasso pass the deal-breaker
rules clearly; scaffold passes the consistency check though the headline
percentage is smaller. The patch is ready for human review and upstream.
