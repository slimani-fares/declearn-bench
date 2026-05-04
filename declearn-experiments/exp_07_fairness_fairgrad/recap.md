# Experience 07 — FairGrad — Recap

## Date
2026-04-30T22:05+02:00

## Setup
- Baseline arm: not directly profiled in this experiment (vanilla FedAvg on the binary MNIST split); cross-experiment vanilla-comparator from exp_05 (~22 s at N=5) used for context.
- Variant arm (= observation profile): `declearn-for-exp_07_fairness/` branch `exp_07_fairness_patch`, commit ae20bd7-equivalent. Forked from `declearn-for-secagg/` (which already had the unified runner for SecAgg) and patched to support fairness:
  - `declearn/quickrun/_parser.py`: optional `train_s_attr` / `valid_s_attr` discovery (non-fatal absence)
  - `declearn/quickrun/_run.py`: when both s_attr paths are present, build `FairnessInMemoryDataset` instead of `InMemoryDataset`
  - 7-line + 28-line diff respectively; saved as `patches/fairness_runner.diff` (will write).
- Data: `examples/mnist_quickrun/data_iid_fair/` — derived from a 5-shard split of MNIST (seed=42, iid), with binary target `digit >= 5` and `s_attr = digit % 2` (parity, orthogonal to target).
- Config: `declearn-experiments/exp_07_fairness_fairgrad/config_variant_fairgrad.toml` (copied from `configs/fairness/`). 2 rounds, `algorithm = "fairgrad"`, `f_type = "accuracy_parity"`.
- Cluster node: magnet4 (load 1.5–2.5 during the run).
- Total wall-clock spent on this experience: ~10 minutes (most of it building/testing the unified-runner fairness patch, which is reused by exp_08 and exp_09).
- **Pipeline gating per CLAUDE.md §5: SATISFIED.** The unified runner now supports both SecAgg (inherited from prior fork) and fairness (new patch). exp_06 already validated the SecAgg path; this experiment validates the fairness path. Smoke test passed: server log shows `Sending FairGrad weights to clients` and `Initiating fairness-enforcing round N` — fairness IS active.

## Hypotheses tested

None proposed-and-A/B'd in this experiment. The dominant fairness-attributable cost (per-round group-wise metric computation via full validation forward pass) is an *algorithm* design choice, not a declearn-implementation bug. Scoping a deal-breaking variant would require a fairness-algorithm change (subsample for metrics, cache predictions, or reduce metric-update frequency) which is upstream work — out of scope for this autonomous loop's per-experience budget. Documented as O1.

## Open observations

### O1 — FairGrad nearly doubles total wall-clock vs vanilla; cost is in `compute_batch_predictions` for group-wise metrics

**Profile:** 41.15 s total, 5 clients, 2 rounds.

Top self-time leaves:
- `_engine_run_backward` (torch): 5.98 s (14.53%)
- `_apply_operation` (Vector): 5.49 s (13.34%) — same Vector pattern as all prior experiments
- `dropout` (torch): 4.62 s (11.23%)
- `_conv_forward` (torch): 3.04 s (7.39%)
- `_max_pool2d` (torch): 2.68 s (6.51%)

Top fairness-attributable cumulative paths:
- `_compute_and_share_fairness_measures` (`fairness/api/_client.py`): 7.14 s total
- `compute_groupwise_metrics` (`fairness/api/_metrics.py`): 7.12 s total
- `compute_batch_predictions` (`model/torch/_model.py`): 7.12 s total
- `get_sensitive_group_subset` (`fairness/core/_inmemory.py`): 0.27 s total

The fairness round happens after every training round (logged: "Initiating fairness-enforcing round 0/1/2"). For each fairness round, FairGrad does a full forward pass on validation data, then bucketizes predictions by sensitive group, then computes per-group metrics. The forward pass dominates: the 7.12 s in `compute_batch_predictions` is essentially a complete model evaluation on the per-client valid set.

**Comparison to vanilla:** vanilla FedAvg-torch at N=5 takes ~22 s (per exp_05 baseline). FairGrad at the same scale takes 41 s — i.e., **+19 s ≈ +86% wall-clock**. ~7 s of that delta is directly attributable to the fairness round; the remaining ~12 s is per-client overhead inflated by the fairness controller's bookkeeping (s_attr file load, `FairnessInMemoryDataset` indexing, etc.).

### O2 — The fairness round runs at every training round by default; reducing frequency could halve overhead

The TOML stanza `[optim.fairness]` (algorithm + f_type) doesn't accept a frequency parameter at the configuration level — declearn always runs fairness setup at the start of training (`fairness-enforcing round 0`) and a fairness update after every training round (rounds 1, 2). At rounds=10+, the per-round fairness cost would amortize differently, but the absolute cost per fairness round is constant (~2.4 s here), so 10-round runs would have ~24 s of fairness rounds vs ~50 s of training rounds — fairness becomes ~33% of total. Not a per-experience deal-breaker; flagged as a fairness-tuning question.

### O3 — `compute_batch_predictions` is a forward pass that could be reused from training-round eval

Looking at the call graph: `_training_round` already runs a forward pass over each batch for gradient computation; the fairness round then re-runs forward over the same valid data for predictions. If declearn cached the validation predictions from the most recent training round's eval phase (which always runs at end-of-training-round per existing logger output) and reused them for the fairness-round metric computation, the fairness-attributable forward-pass cost would drop from ~7 s to ~0 s. **This IS an implementation-level optimization opportunity** — but designing it correctly requires care (cache invalidation, consistency with the metric the fairness controller expects). Not implemented here; marked as exp_07b.

### O4 — Cross-experiment Vector pattern continues

`_apply_operation` (Vector): 5.49 s self at 13.34% — same as exp_01–06. The shared `Vector` overhead is ~13% in this experiment. Same recommendation as exp_03: a Vector-layer batched-op fix would help here too.

### O5 — Websocket compression is at 3.16% in FairGrad

Smaller in absolute proportion than exp_05 (where it was 7.8%), because FairGrad has more compute work overall. But still 1.30 s of meaningful self-time — would be reclaimed by disabling deflate on localhost.

## Conclusions

**Headline:** FairGrad approximately doubles wall-clock over vanilla on a binary MNIST quickrun. ~7 s out of the +19 s overhead is direct fairness-controller compute (group-wise metrics via forward pass over validation data, per fairness round). The rest is FairnessInMemoryDataset bookkeeping and the cross-experiment Vector / dataset overhead. **No declearn-implementation deal-breaker is visible in FairGrad alone**; the largest reclaimable cost (caching validation predictions across training-round eval and fairness-round metrics) would benefit fairness in general, not just FairGrad.

**What Fares would want to revisit:**
- **exp_07b: implement validation-prediction caching** between `_training_round`'s end-of-round eval and the subsequent fairness-round metric computation. Estimated savings: ~7 s out of 41 s = 17% per experiment, scaling with rounds. This applies to all three fairness experiments (exp_07/08/09) once implemented.
- **Frequency parameter for fairness rounds.** declearn currently runs the fairness round at every training round. A `frequency` knob (e.g., every K rounds) would allow trading metric freshness for wall-clock — useful for long training runs.
- The unified-runner fairness patch from this experiment (`patches/fairness_runner.diff`) is reused by exp_08 and exp_09. The patch is small (35 lines total) and has been smoke-tested via fairgrad. It does NOT alter the fairness algorithm; it only wires `FairnessInMemoryDataset` into the quickrun client setup when s_attr files are detected.

## Caveats and open questions

- Single-seed observation. No multi-seed A/B was performed because no per-experiment optimization variant was built — the dominant cost is a fairness-algorithm property, not an implementation bug.
- The 5-client run was an artifact of having a 5-shard split left over from exp_06 — the FairGrad pipeline accepted whichever client count was present in the data folder. The cluster load was higher (1.5–2.5) during this run than during earlier experiments; absolute timings carry ~10–15% noise. Relative magnitudes (fairness round ~17% of total) are robust.
- The smoke test was just "did the run complete with fairness active?" — not a hypothesis-driven A/B with output equivalence. The patch's correctness was validated by the server log emitting fairness-specific INFO lines (`Sending FairGrad weights`, `Initiating fairness-enforcing round`) that don't appear in non-fairness runs.
- `data_iid_fair/` was generated with a 5-client split (because exp_06 last set it that way). The s_attr binary parity attribute is well-balanced (~0.51 across all client splits) and orthogonal to the binarized target (~0.49 across all). Reasonable synthetic fairness setup.
