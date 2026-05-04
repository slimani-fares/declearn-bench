# Experience 08 — FairBatch — Recap

## Date
2026-04-30T22:08+02:00

## Setup
- Variant arm (= observation profile): same `declearn-for-exp_07_fairness/` fork as exp_07 — the unified runner fairness patch is reused.
- Config: `declearn-experiments/exp_08_fairness_fairbatch/config_variant_fairbatch.toml` (copied from `configs/fairness/`). 2 rounds, `algorithm = "fairbatch"`, `f_type = "demographic_parity"`.
- Data: `examples/mnist_quickrun/data_iid_fair/` (5 clients, binary target = digit≥5, s_attr = parity).
- Cluster node: magnet4 (load 0.9–1.4 during the run).
- Total wall-clock spent on this experience: ~3 minutes (no patch work — pipe was already warm from exp_07).

## Hypotheses tested

None proposed-and-A/B'd. FairBatch shows the same overall cost shape as FairGrad (exp_07): the dominant cost is fairness-controller compute (per-round group-wise metric computation + sampling-probability updates), which is an algorithm choice rather than an implementation bug. The same suggested optimization (cache validation predictions for reuse across training-round eval and fairness-round metrics) would benefit both algorithms.

## Open observations

### O1 — FairBatch wall-clock is functionally identical to FairGrad: 41.01 s vs 41.15 s

| metric | FairBatch (exp_08) | FairGrad (exp_07) | vanilla N=5 (exp_05) |
|---|---|---|---|
| total wall-clock | 41.01 s | 41.15 s | 22.09 s |
| `_engine_run_backward` self | 6.12 s (14.92%) | 5.98 s (14.53%) | 3.35 s (15.16%) |
| `Vector._apply_operation` self | 5.86 s (14.29%) | 5.49 s (13.34%) | 3.75 s (16.96%) |
| `dropout` self | 4.65 s (11.34%) | 4.62 s (11.23%) | 2.73 s (12.36%) |
| `_max_pool2d` self | 2.73 s (6.66%) | 2.68 s (6.51%) | — |
| `_conv_forward` self | 2.34 s (5.71%) | 3.04 s (7.39%) | 0.71 s (3.21%) |
| `permessage_deflate.encode` self | 1.16 s (2.83%) | 1.30 s (3.16%) | 1.31 s (5.93%) |
| `get_sensitive_group_subset` self | 0.62 s (1.51%) | 0.27 s (0.66%) | n/a |

Three fairness-enforcing rounds were observed in the log (rounds 0, 1, 2 — same shape as FairGrad). FairBatch exchanges *sampling probabilities* (logged: "Sending FairBatch sampling probabilities to clients") instead of FairGrad's gradient-correction weights, but the per-round cost is similar.

### O2 — FairBatch's `get_sensitive_group_subset` is 2.3x more expensive than in FairGrad

The `get_sensitive_group_subset` call (at `fairness/core/_inmemory.py`) carries 0.62 s self in FairBatch vs 0.27 s in FairGrad. This is FairBatch reweighing samples by sensitive group at every batch — not a bug, just a fact of the algorithm. At larger datasets this scales linearly with sample count and worth flagging if it ever becomes a top-5 hotspot. Currently 1.5% of total — well below noise floor.

### O3 — Final-round accuracy is much better than FairGrad

FairBatch: server-averaged accuracy 0.7772 at round 2.
FairGrad: server-averaged accuracy 0.5103 at round 2.

This isn't a profiling result — it reflects how well each algorithm converges on the binary MNIST task. Useful as a sanity check that the patch is wired correctly: FairBatch is well-behaved, FairGrad takes more rounds to converge. Not a per-algorithm criticism.

### O4 — Same cross-experiment conclusions as exp_07

- Vector dispatch overhead: 14.29% self in FairBatch — consistent with exp_01–07 cross-experiment 13–17% pattern.
- Websocket compression at 2.83% — same opportunity-cost as elsewhere.
- Forward-pass-on-validation-data cost would be reclaimable by caching predictions across training-round eval and fairness-round metrics. Same exp_07b candidate.

## Conclusions

**Headline:** FairBatch's profile shape is essentially indistinguishable from FairGrad (within ~0.3% on every major hotspot). The +86% wall-clock vs vanilla is a fairness-controller infrastructure cost (full forward pass per fairness round + per-batch group reweighing) shared across both algorithms. The exp_07 conclusion stands: no FairBatch-implementation deal-breaker; the largest reclaimable cost is *predictions caching*, applicable to both.

**What Fares would want to revisit:**
- exp_07b (validation-prediction caching) is the unified intervention for exp_07/08/09. Implementing it once and re-running all three fairness experiments would test whether the ~7 s fairness-round savings actually materialize.
- The fact that FairGrad and FairBatch produce nearly identical profiles despite very different algorithms suggests the bottleneck is the fairness *infrastructure* (controller, dataset wrapper, message exchange) rather than algorithm-specific compute. A future optimization should target that infrastructure rather than per-algorithm tuning.

## Caveats and open questions

- Single-seed observation. Same as exp_07.
- 5-client run (legacy from exp_06's data split). Same as exp_07 — paired comparison to that experiment is fair because both used the same data split.
- Cluster load was lower than during exp_07 (~1.0 vs ~1.5–2.5). Both arms produced ~41 s nonetheless; the load drop didn't translate to a wall-clock change, suggesting either the work is CPU-bound enough to saturate even modest contention, OR the noise contribution is small at this scale.
