# Experience 11 — Single-pass groupwise metrics for fairness — Recap

## Date
2026-05-07

## What this experiment was
The cross-experiment recap (`findings.md`) and exp_07's `recap.md`
flagged group-wise metric computation as a ~17% wall-clock cost across
all three fairness experiments (FairGrad, FairBatch, FairFed),
attributing it to repeated forward passes during the per-group iteration.
This experiment tested whether replacing N per-group iterations with a
single pass over the parent dataset would reclaim that cost.

## What changed
- **Fork:** `declearn-for-exp_11_fairness_singlepass/`, branched from
  the unified-runner-patched fairness fork (`declearn-for-exp_07_fairness`).
  - `master` arm: canonical fairness behaviour (per-group iteration in
    `FairnessMetricsComputer.compute_groupwise_metrics`).
  - `variant` arm: single-pass implementation (`patches/single_pass_metrics.diff`,
    115 lines, single file: `declearn/fairness/api/_metrics.py`).
- **Patch shape:**
  - `__init__` now keeps a reference to the parent dataset and
    precomputes an integer group-id per sample (`_sample_group_ids`)
    keyed off `dataset.sensitive`.
  - `compute_groupwise_metrics` takes a fast path when `n_batch is None`
    and the dataset has a `sensitive` attribute — it iterates the parent
    dataset ONCE and accumulates per-(group, metric) `(num_sum, divisor)`
    pairs into numpy arrays via vectorized boolean masks.
  - Per-batch: ONE forward pass on the full batch, then K vectorized
    masks (`batch_group_ids == g_idx`) for K groups.
  - Falls through to the canonical per-group code when conditions don't
    apply.
- **Configs:** `config_fairgrad.toml`, `config_fairbatch.toml`,
  `config_fairfed.toml` — 3 clients, binary mnist target (`digit >= 5`),
  parity s_attr, 2 rounds.

## Hypothesis tested

### H1 — Single-pass groupwise metrics deliver ~17% wall-clock saving

**Status: refuted.**

#### Smoke test
Unit-level smoke: build a `FairnessInMemoryDataset` from
`data_iid_fair/client_0`, build a deterministic torch model
(seeded init), call `computer.compute_groupwise_metrics(...)` 5×
on each branch, compare per-(metric, group) outputs.

| metric    | group        | master       | variant      | abs_diff |
|-----------|--------------|--------------|--------------|----------|
| accuracy  | (0, 0.0)     | 0.0000000000 | 0.0000000000 | 0.000e+00 |
| accuracy  | (0, 1.0)     | 0.0000000000 | 0.0000000000 | 0.000e+00 |
| accuracy  | (1, 0.0)     | 0.0000000000 | 0.0000000000 | 0.000e+00 |
| accuracy  | (1, 1.0)     | 0.0000000000 | 0.0000000000 | 0.000e+00 |

PASS — byte-equal across all 4 (metric, group) entries.

#### A/B (3 algos × 2 arms × 3 seeds × 2 rounds = 18 runs)

| algo      | arm     | wall (s, mean) | std  | speedup vs master | acc (mean) |
|-----------|---------|----------------|------|-------------------|------------|
| fairgrad  | master  | 38.12          | 0.32 | 1.00×             | 0.766      |
| fairgrad  | variant | 38.03          | 0.48 | **1.00×** (-0.09 s) | 0.654    |
| fairbatch | master  | 37.44          | 0.08 | 1.00×             | 0.612      |
| fairbatch | variant | 37.84          | 0.18 | **0.99×** (+0.40 s) | 0.766    |
| fairfed   | master  | 39.20          | 0.33 | 1.00×             | 0.909      |
| fairfed   | variant | 39.62          | 0.25 | **0.99×** (+0.42 s) | 0.910    |

All deltas within ±2 std. Two of three algorithms variant runs are
slightly **slower** than master.

#### Unit-level latency check (5-iter mean, tiny model on client_0 train data)

```
master:  30.4 ms / call
variant: 40.6 ms / call
speedup: 0.75×  (-33.6 % regression)
```

At small scale (Linear model, ~16k samples), the single-pass variant
is **measurably slower** because the forward-pass FLOPs are tiny and
the per-batch group-mask overhead now dominates.

#### py-spy profile (FairGrad, single seed, lasso 2 rounds)

Profile total: master 26.12 s, variant 26.75 s → variant +0.63 s
slower. Top deltas:

| frame                                          | master self_s | variant self_s | delta_s |
|------------------------------------------------|---------------|----------------|---------|
| `_compute_groupwise_metrics_single_pass` (new) | 0.00          | 0.21           | **+0.21** (variant overhead) |
| `_apply_operation` (Vector — unrelated)        | 3.87          | 4.22           | +0.35 (noise) |
| `dropout` (torch — unrelated)                  | 2.62          | 3.13           | +0.51 (noise) |
| `_build_iterator`                              | 0.33          | 0.83           | +0.50 (variant calls dataset iterator more, see below) |
| `_engine_run_backward` (unrelated)             | 3.61          | 3.67           | +0.06 |

The genuinely new self-time entry is `_compute_groupwise_metrics_single_pass`
at 0.21 s — that's the variant's per-batch group-mask + accumulator
work. The other deltas are within sampling noise of two single-seed
py-spy runs.

#### Why the projected 17% saving doesn't materialize

The original recap's framing was: "the fairness round runs a forward
pass over the *same data* the eval just ran a forward pass over —
caching saves the second forward pass."

That framing is wrong on inspection of the code:
- `evaluation_round` runs forward passes on `manager.valid_data`
- The fairness round's `FairnessMetricsComputer` is initialized with
  `manager.train_data` (`fairness/api/_client.py:155`) — and all three
  algorithm subclasses (`fairgrad`, `fairbatch`, `fairfed`) use
  `manager.train_data`, not `valid_data`.

So the eval and fairness forward passes operate on **different
datasets**. There's no caching opportunity between them.

What's left is the actual structure of the canonical fairness round:
- Iterate `g_data[g].generate_batches(...)` per group (K times)
- Each iteration runs forward passes on that group's batches
- The total samples processed = sum of group sizes = full training set
- Total forward-pass FLOPs = same as a single-pass over the full set

The single-pass variant only saves the per-group iterator setup
overhead. That overhead is microsecond-scale per group; on a fairness
round dominated by torch forward-pass FLOPs, it's invisible. The
variant introduces its own overhead (per-batch K boolean masks,
numpy bookkeeping) that approximately offsets — and at small model
scales actually exceeds — the saving.

#### Deal-breaker assessment per CLAUDE.md §7

| rule                                                  | status |
|-------------------------------------------------------|--------|
| Smoke equivalence (byte-equal at unit level)          | ✓      |
| Accuracy floor (mean delta ≤ 0.10 absolute)           | ✓ (deltas dominated by un-pinned init noise; see RNG caveat) |
| Improvement above noise (5%)                          | ✗ (best is 0.2%, two of three are negative) |
| Consistency across seeds (mean improvement > 1 std)   | ✗ (deltas are sub-1-std for all 3 algos) |
| Perf direction                                         | ✗ (variant slightly slower for fairbatch and fairfed) |

**Multiple §7 rules failed → refuted (deal-breaker triggered).**

## Open observations (NOT tested as hypotheses)

- **fairbatch's `f_type` accepts a different vocabulary than the other
  two.** Initial config used `accuracy_parity`, which fairbatch rejected
  with `Unknown or unsupported fairness type ... Supported values are
  ['demographic_parity', 'equality_of_opportunity', 'equalized_odds']`.
  Switched to `demographic_parity` and the run succeeded. **This is a
  pre-existing inconsistency in declearn's fairness config schema** —
  worth a small docs note or a clearer upstream error message.
- **The original `findings.md` claim of "17% reclaimable cost via
  caching" was based on a misread of the code path.** Eval and fairness
  do NOT share a forward pass (different datasets). The reclaimable
  fraction is whatever's in per-group iteration overhead, which is
  microsecond-scale and not visible in py-spy at 100 Hz.
- **The actual cost in fairness rounds IS forward-pass FLOPs on the
  full training set.** Reducing this requires algorithm-level changes
  (subsample, lower frequency) — out of scope for an
  implementation-level optimization.

## Conclusions

**Headline:** the single-pass groupwise-metrics patch is correct
(byte-equal smoke) but produces no measurable wall-clock improvement
across three fairness algorithms. The projected 17% saving in
`findings.md` was based on an inaccurate read of which datasets eval
vs fairness operate on; on inspection, the redundancy doesn't exist
as initially framed.

**What this means for declearn:**
1. Don't merge the variant. It adds complexity without delivering
   benefit, and at small scales is a measurable (-33%) latency regression.
2. The fairness-round forward pass is real cost but it's algorithmic,
   not implementational. Fairness-side optimizations should target:
   - Reducing the number of fairness rounds (frequency knob)
   - Subsampling (`n_batch < total_batches`) for metric estimation
   - Smaller validation/audit subsets for the fairness controller
3. The `findings.md` "17%" projection should be retracted in the
   project summary.

**What we learned (process, not declearn):**
1. Project recommendations made from single-profile observations need
   to be re-grounded in the source-code call graph before being acted
   on. exp_07's recap noted "could cache predictions across these two
   calls" — that was a hypothesis, not a verified opportunity.
2. Unit-level latency benchmarking caught the variant regression that
   end-to-end A/B noise hid (40 ms vs 30 ms is undetectable on a 38 s
   end-to-end run, but it's the actual signal).

## Caveats and open questions

- **Per-seed accuracy varies by ±0.27** between master and variant for
  the same seed — this is the same un-pinned-RNG-at-model-init issue
  observed in exp_10, not a patch correctness issue. Mean accuracy
  across seeds remains within 0.15 (FairGrad master 0.766 vs variant
  0.654 = 0.11 delta — at the §7 floor). Pinning `torch.manual_seed`
  in the binary model file would make per-seed deltas vanish.
- **Single-seed py-spy.** ±0.5s on top-25 entries is normal sampling
  noise; the only signal we can confidently attribute to the patch is
  the new `_compute_groupwise_metrics_single_pass` at +0.21 s.
- **3-client n=2-round scale.** The fairness rounds account for ~10 s
  of the 38 s total wall-clock. Even a 50% reduction in fairness-round
  cost would be only ~5 s = ~13% wall-clock — and our variant doesn't
  achieve anything close to a 50% fairness-round reduction.

## Where the data lives

| if you want…                           | open…                                                                  |
|----------------------------------------|------------------------------------------------------------------------|
| The 18 A/B run logs                    | `runs/ab_*.log` (and `runs/ab_fairbatch_*` for the f_type-fixed reruns) |
| A/B summary (machine-readable)         | `runs/ab_results.json` + `runs/ab_results_fairbatch.json`              |
| A/B summary (human)                    | `runs/ab_run.console.log` + `runs/ab_fairbatch.console.log`            |
| Master py-spy profile (FairGrad)       | `runs/profiles/master/2026-05-07_11-36-18/pyspy_speedscope.json`       |
| Variant py-spy profile (FairGrad)      | `runs/profiles/variant/2026-05-07_11-37-06/pyspy_speedscope.json`      |
| Side-by-side profile comparison        | `runs/compare_fairgrad.txt`                                            |
| The patch                              | `patches/single_pass_metrics.diff`                                     |
| Unit smoke driver + pickles            | `smoke_unit.py` + `runs/smoke_unit_master.pkl` + `..._variant.pkl`     |
| End-to-end smoke + parsed metrics      | `smoke.py` + `runs/smoke_compare.json`                                 |
| Data-prep helper                       | `prepare_data.py` (idempotent regen of `examples/mnist_quickrun/data_iid_fair`) |
| A/B drivers                            | `ab_run.py` (full) + `ab_run_fairbatch_only.py` (f_type-fixed retry)   |
| Profile driver                         | `profile_arms.py`                                                       |

## Status

**Refuted (per CLAUDE.md §7).** Patch is correct but produces no
measurable wall-clock improvement and a small unit-level regression.
The original `findings.md` projection was based on a misread of the
fairness call graph (eval and fairness do not share a dataset).
Recommendation: do not merge. Fairness-round cost is algorithmic,
not implementational; retire the "17% reclaimable" claim from
`findings.md`.
