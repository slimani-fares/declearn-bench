# Experience 02 — Ridge regularizer — Recap

## Date
2026-04-30T21:28+02:00

## Setup
- Baseline: declearn 2.8.0 canonical, vanilla FedAvg-torch profile from `declearn-experiments/_setup/baselines/baseline_vanilla_2r/2026-04-30_21-23-55/`
- Variant fork(s): none
- Data split: `examples/mnist_quickrun/data_iid/` (n_shards=2, seed=42, iid)
- Cluster node: magnet4 (load 0.49 → 0.64 over the run; still under the 4.0 floor)
- Total wall-clock spent on this experience: ~3 minutes

## Hypotheses tested

None. Ridge's measured overhead vs vanilla baseline is +0.47 s = +2.2%, which sits at/below the §9 5% noise floor. There is no candidate optimization that would survive A/B confirmation at this magnitude.

## Open observations

### O1 — Ridge adds ~2% wall-clock; cost is one `_apply_operation` + one `get_weights` per step

**Comparison:**
- Vanilla baseline (2c, 2r): 20.95 s total weight
- Ridge variant (2c, 2r):    21.42 s total weight
- Δ = +0.47 s = +2.2%

Top deltas in self-time, ridge − vanilla:
- `Vector._apply_operation`: +0.42 s
- `_model.get_weights`:      +0.32 s (NEW)
- All torch ops:             ≈ ±0.2 s (run-to-run noise)

`RidgeRegularizer.run` (declearn/optimizer/regularizers/_base.py:123-129) is structurally simpler than lasso:
```python
correct = 2 * self.alpha * weights   # 1 _apply_operation (scalar mul)
return gradients + correct            # 1 _apply_operation (vector add)
```

No `weights.sign()` call → no `apply_func` overhead → ridge avoids the +1.22 s `apply_func` cost that dominated lasso (exp_01). The remaining cost is the same generic pattern: the regularizer is invoked per training step and each Vector op iterates over parameter tensors in Python.

### O2 — Confirms the lasso finding generalizes: regularizer overhead is regularizer-pattern-generic

The +0.42 s `_apply_operation` delta in ridge nearly matches the +0.40 s in lasso. The +0.32 s `get_weights` matches lasso's +0.34 s. This strengthens exp_01's conclusion: regularizer cost is dominated by Python-level Vector dispatch and per-step `get_weights`, neither of which is regularizer-specific. The lasso-vs-ridge gap (+12.5% vs +2.2%) is entirely explained by lasso's extra `apply_func(torch.sign)` Vector op.

### O3 — Same dropout / websocket-deflate background as exp_01

Dropout (~11.5% self) and `permessage_deflate.encode` (~2.5%) appear at near-identical magnitudes in both ridge and vanilla profiles. They are not regularizer-attributable.

## Conclusions

Ridge regularization in declearn carries ~2% wall-clock overhead at this scale, dominated by two declearn Python-level overheads (`_apply_operation` + `get_weights`) common to all regularizers. Below the noise floor for a per-experience optimization. The cleanest takeaway: **lasso's much larger overhead is the specific cost of the `sign()` Vector op, not regularization in general.**

What Fares would want to revisit: at higher scales (more parameters, more SGD steps per round, more clients) ridge's 2% may grow proportionally with `_apply_operation` calls — worth re-measuring at n_clients=10 and rounds=10 to confirm the overhead doesn't compound.

## Caveats and open questions

- 2c × 2r × 1 seed; not statistically validated.
- Cluster load was rising (0.49 → 0.64) during the run. Probably another tenant booting; magnet4 is mostly idle, so noise is small but not zero.
- The −0.26 s drop on `_conv_forward` and −0.19 s on `_max_pool2d` between vanilla and ridge is likely OpenMP thread-affinity variance, not a real ridge-induced speedup. Reconciliation: total +0.47 s is well below the 5% noise floor; treating as noise.
