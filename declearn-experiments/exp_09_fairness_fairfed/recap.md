# Experience 09 — FairFed — Recap

## Date
2026-04-30T22:11+02:00

## Setup
- Variant arm (= observation profile): same `declearn-for-exp_07_fairness/` fork as exp_07/08.
- Config: `declearn-experiments/exp_09_fairness_fairfed/config_variant_fairfed.toml`. 2 rounds, `algorithm = "fairfed"`, `f_type = "accuracy_parity"`, `strict = true`, `beta = 1.0`.
- Data: same `examples/mnist_quickrun/data_iid_fair/` as exp_07/08 (5 clients, binary target, parity s_attr).
- Cluster node: magnet4 (load 1.4–1.6).
- Total wall-clock spent on this experience: ~3 minutes.

## Hypotheses tested

None. Same reasoning as exp_07 and exp_08: the dominant fairness costs are infrastructure-shared. Implementing the cross-experiment "exp_07b: cache validation predictions" optimization is the unified intervention; an FairFed-specific A/B without that broader fix would either be redundant or testing an algorithm-design choice rather than a declearn-implementation issue.

## Open observations

### O1 — FairFed is the lightest fairness algorithm AND has the best accuracy

| algorithm | total wall-clock | server-avg accuracy at round 2 |
|---|---|---|
| FairGrad (exp_07)   | 41.15 s | 0.5103 |
| FairBatch (exp_08)  | 41.01 s | 0.7772 |
| FairFed (exp_09)    | 36.92 s | **0.9343** |

FairFed is ~10% faster than the other two and converges to 93% accuracy in just 2 rounds. The likely reason is structural: FairFed's fairness enforcement is primarily a server-side reweighting of client updates during aggregation, rather than gradient correction (FairGrad, applied per-step) or batch reweighting (FairBatch, applied per-step). Clients in FairFed do straight FedAvg; the fairness work happens at aggregation time. Less per-step client work → less wall-clock.

### O2 — Profile shape is otherwise consistent with exp_07/08

Top hotspots, FairFed:
- `Vector._apply_operation`: 5.60 s (15.17%) — same Vector pattern as all prior experiments
- `_engine_run_backward`: 4.75 s (12.87%)
- `dropout`: 4.11 s (11.13%)
- `_conv_forward`: 2.57 s (6.96%)
- `_max_pool2d`: 2.47 s (6.69%)
- `permessage_deflate.encode`: 1.08 s (2.93%)

Three fairness-enforcing rounds were observed in the log, same shape as the other two. The fairness-attributable cumulative cost is bounded by the difference 36.92 − 22.09 (vanilla N=5) = **+14.83 s ≈ +67% wall-clock vs vanilla** — less than FairGrad's +86% and FairBatch's +86%, consistent with FairFed doing less per-step client-side work.

### O3 — `get_sensitive_group_subset` is essentially absent in FairFed (≪ 0.1 s)

Unlike FairBatch (0.62 s) and FairGrad (0.27 s), FairFed has no measurable `get_sensitive_group_subset` cost in this profile. This confirms the structural reading in O1: FairFed doesn't need to bucketize samples per training step, only at aggregation time. Server-side accounting hides this cost in the (per-round) aggregation step rather than spreading it across the per-step training loop.

### O4 — Vector + websocket compression patterns repeat for the 9th time

This is the 9th experiment in a row to show:
- `Vector._apply_operation` at 13–17% self-time
- `permessage_deflate.encode` at 2–8% self-time

These are the two most consistent cross-experiment hotspots in the entire loop. They're independent of the algorithm being tested (regularizer, DP, SCAFFOLD, SecAgg-batched, fairness-anything). The cross-experiment recommendation now has nine pieces of supporting evidence.

## Conclusions

**Headline:** FairFed is the cheapest of the three fairness algorithms tested AND converges fastest (best accuracy at round 2). The fairness-controller infrastructure cost (per-round group metric computation) is smaller in FairFed because the algorithm's fairness work is server-side aggregation rather than per-step client work.

**What Fares would want to revisit:**
- The same exp_07b candidate (predictions caching) would benefit FairFed less than the other two, because FairFed already amortizes more work into the aggregation step. A targeted FairFed-specific optimization is harder to identify; the algorithm seems already well-aligned with declearn's per-round structure.
- **Cross-experiment finding (independent of fairness):** the consistent +13–17% Vector dispatch overhead AND +2–8% websocket compression appear in EVERY one of the 9 experiments. A single targeted experiment on these two infrastructure costs (foreach-Vector and disable-deflate-on-localhost) would benefit every algorithm declearn supports.

## Caveats and open questions

- Single-seed observation. Same as exp_07 and exp_08.
- The wall-clock hierarchy (FairFed < FairBatch ≈ FairGrad) may reverse at higher round counts if FairFed's server-side aggregation cost grows non-linearly with rounds or with model size. Worth verifying at rounds=10 or N=20.
- 5-client data split (legacy from exp_06). Comparison with exp_07/08 is fair (same data); comparison with cross-experiment "vanilla N=5" baseline (22.09 s from exp_05) uses the same N but a slightly different data folder (`data_iid` vs `data_iid_fair` differ in target binarization but not in input features). Reasonable comparator.
