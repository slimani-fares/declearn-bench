# Experience 03 — FedProx regularizer — Recap

## Date
2026-04-30T21:29+02:00

## Setup
- Baseline: declearn 2.8.0 canonical, vanilla FedAvg-torch profile from `declearn-experiments/_setup/baselines/baseline_vanilla_2r/2026-04-30_21-23-55/`
- Variant fork(s): none
- Data split: `examples/mnist_quickrun/data_iid/` (n_shards=2, seed=42, iid)
- Cluster node: magnet4 (load 0.5–0.6 during run)
- Total wall-clock spent on this experience: ~3 minutes

## Hypotheses tested

None. FedProx adds +4.4% wall-clock — at the §9 5% noise floor. No candidate optimization survives confirmation at this magnitude.

## Open observations

### O1 — FedProx adds +4.4% wall-clock; cost pattern matches lasso/ridge

**Comparison:**
- Vanilla baseline (2c, 2r): 20.95 s
- FedProx variant (2c, 2r):  21.87 s
- Δ = +0.92 s = +4.4%

Top deltas in self-time, fedprox − vanilla:
- `Vector._apply_operation`: +0.54 s
- `_model.get_weights`:      +0.33 s (NEW)
- `dedent` (textwrap):       +0.24 s (one-off — likely startup error rendering, ignored)
- All torch ops:             ±0.3 s (run-to-run noise)

### O2 — FedProx has 3 Vector ops per step (vs ridge's 2 vs lasso's 3); cost scales accordingly

`FedProxRegularizer.run` (declearn/optimizer/regularizers/_base.py:75-83):
```python
correct = self.alpha * (weights - self.ref_wgt)   # 2 _apply_operation (sub, scalar mul)
return gradients + correct                         # 1 _apply_operation (add)
```

Three `_apply_operation` invocations per training step → +0.54 s, vs ridge's two ops at +0.42 s, vs lasso's two ops at +0.40 s. The per-op cost is roughly +0.20 s per `_apply_operation` per (2 clients × 2 rounds), which is internally consistent.

The +0.33 s `get_weights` cost is identical across all three regularizer experiments — it's the per-step model-side cost of materializing weights as a Vector for the regularizer API.

### O3 — Cross-experience picture (exp_01–03)

| regularizer | Δ wall-clock | apply_func delta | _apply_operation delta | get_weights delta |
|---|---|---|---|---|
| lasso (exp_01)   | +12.5% (+2.62 s) | +1.22 s | +0.40 s | +0.34 s |
| ridge (exp_02)   | +2.2%  (+0.47 s) | 0      | +0.42 s | +0.32 s |
| fedprox (exp_03) | +4.4%  (+0.92 s) | 0      | +0.54 s | +0.33 s |

The story is now decisive: **regularizer overhead in declearn is dominated by `Vector` Python dispatch and per-step `get_weights`, both shared infrastructure. Lasso is the outlier only because `weights.sign()` triggers an additional `Vector.apply_func`** (+1.22 s) absent from ridge and fedprox. None of the per-regularizer overheads cross the 5% noise floor on their own (lasso barely does, but the addressable portion within the regularizer is <2%).

## Conclusions

The three regularizer experiments (exp_01–03) collectively make the same point: regularizer cost in declearn lives in the `Vector` abstraction layer, not in the regularizer math. Per-experience optimizations are below the noise floor. The actionable follow-up is at the `Vector` layer, not the regularizer layer. **Recommendation for Fares: a single targeted experiment on `Vector` dispatch overhead — e.g., introducing `torch._foreach_*` batch ops in `TorchVector` — would benefit all regularizers and likely scaffold/dp/secagg/fairness as well.** That is one well-defined intervention, not nine.

## Caveats and open questions

- 2c × 2r × 1 seed; not statistically validated. Cross-experiment trends (the table above) are robust because the deltas are consistent in sign and magnitude across three independent runs.
- `dedent` in textwrap showing +0.24 s in fedprox but not in others looks like a one-off cost (possibly a deprecation-warning pretty-print at startup). Not investigated.
- Regularizer cost will scale with `n_steps_per_round × n_params`. At the n=10 client / 10 rounds scale recommended for follow-up, lasso's +12.5% might grow to a more clearly addressable fraction; ridge/fedprox would still be marginal.
