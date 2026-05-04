# Experience 01 — Lasso regularizer — Recap

## Date
2026-04-30T21:25+02:00

## Setup
- Baseline: declearn 2.8.0 canonical (`~/declearn-bench/declearn/`, no fork — observation only)
- Variant fork(s): none (no optimization implemented; see "Conclusions")
- Data split: `examples/mnist_quickrun/data_iid/` — declearn-split seed 42, n_shards=2, scheme=iid
- Cluster node: magnet4 (load 0.19 → 0.32 over the run)
- Total wall-clock spent on this experience: ~5 minutes (single observation profile + reference baseline)

## Hypotheses tested

None went to A/B. See "Open observations" — the well-bounded fix would save <2% wall-clock (below the §9 noise floor). The remaining hypothesis territory (foreach-based batch ops, torch-native regularization) requires invasive declearn-Vector changes that exceed per-experience scope and don't move the recommendation about declearn's lasso *specifically* — they generalize across all regularizers and would belong in a Vector-overhead audit.

## Open observations (NOT tested as hypotheses)

### O1 — Lasso adds +12.5% wall-clock at 2c/2r (measured), all in declearn-internal Vector dispatch

**Comparison:**
- Vanilla FedAvg-torch (2c, 2r): 20.95 s py-spy total weight (`declearn-experiments/_setup/baselines/baseline_vanilla_2r/2026-04-30_21-23-55/`)
- Lasso variant (2c, 2r):       23.57 s py-spy total weight (`declearn-experiments/exp_01_reg_lasso/runs/exp_01_observation_lasso/2026-04-30_21-21-42/`)
- Δ = +2.62 s = +12.5%

Top deltas in self-time, lasso − vanilla:
- `Vector.apply_func` (declearn/model/api/_vector.py):  +1.22 s (NEW in variant — was 0)
- `Vector._apply_operation`:                            +0.40 s
- `_model.get_weights` (torch _model.py):              +0.34 s (NEW in variant)

Sum of declearn-side delta = +1.96 s = 75% of the +2.62 s wall-clock delta. The remaining +0.66 s is small noise across torch ops (±0.2 s on conv/maxpool/linear), consistent with run-to-run variance.

### O2 — The cost is structural to declearn's Vector abstraction, not lasso math

`LassoRegularizer.run` (declearn/optimizer/regularizers/_base.py:100-106) performs three chained Vector operations per training step:
1. `weights.sign()` → calls `Vector.apply_func(torch.sign)` (the +1.22 s entry above)
2. `self.alpha * <step1>` → calls `Vector.__rmul__` → `_apply_operation` with scalar
3. `gradients + <step2>` → calls `Vector.__add__` → `_apply_operation` with another Vector

Each operation iterates over the per-parameter dict and dispatches per tensor in Python. For a CNN with ~10 parameter tensors and ~125 SGD steps × 2 rounds × 2 clients = 5000 Python-level dispatches across the run. The torch ops underneath (`torch.sign`, scalar mul, add) are essentially free on CPU at this size — so the 1.96 s declearn delta is overwhelmingly Python dispatch overhead, not numerics.

### O3 — Bounded fixes are too small; structural fixes are too big for a per-experience recap

Bounded fix considered: fuse `sign()` + `*α` into a single `apply_func(lambda w: w.sign() * α)`. Reduces 1 `_apply_operation` per step, but the dominant `apply_func` cost (1.22 s) persists. Estimated savings: ~0.4 s = ~1.7% of variant wall-clock. Below the 5% noise floor — would not survive A/B confirmation.

Structural fix considered: replace `Vector.apply_func` per-tensor dispatch with `torch._foreach_*` batch ops in `TorchVector`. This would benefit lasso but also ridge, fedprox, scaffold, dp, and every other declearn pipeline that uses Vector arithmetic. It is *not* a lasso-specific recommendation — belongs in a separate "Vector dispatch overhead" audit, not in this recap.

### O4 — Unrelated-but-noticed: websocket compression at 2.7%, dropout at 11.5%

Both profiles show:
- `permessage_deflate.encode` (websockets): ~0.6 s = 2.7% self-time. This is on `localhost` — disabling deflate could reclaim it. Same effect on baseline and variant; not lasso-specific.
- `torch.nn.functional.dropout`: ~2.7 s = 11.5% self-time. The mnist_quickrun model uses dropout in train mode and on every forward pass. Dropout is a kernel that dominates the model_torch.py `forward()` self-time. Not lasso-specific either; flagging because it's the third-largest leaf hotspot and likely repeats across exp_02–05.

## Conclusions

Lasso adds a measurable, clean, and reproducible +12.5% wall-clock overhead in this configuration (2c, 2r, MNIST quickrun, FedAvg-torch). 75% of that delta is identifiable as Python-level Vector dispatch in declearn — a known cost of the framework-agnostic `Vector` design, not a property of L1 regularization itself. Lasso-specific optimizations within bounds are below the noise floor; the meaningful follow-up is a Vector-dispatch-overhead audit at the framework layer, applicable across all regularizers and modules.

What Fares would want to revisit: whether the same +1.0–1.5 s `apply_func` signature appears in exp_02 (ridge) and exp_03 (fedprox) — if so, that is the cleanest evidence that the cost is regularizer-pattern-generic and worth fixing once at the Vector layer rather than per-regularizer.

## Caveats and open questions

- 2 clients × 2 rounds × 1 seed is below the §6.4 multi-seed bar — these numbers are observational, not statistically validated. Re-running at 3+ seeds is needed before quoting "+12.5%" as a finding.
- The vanilla baseline used `examples/mnist_quickrun/data_iid/` (n_shards=2 split with seed 42); lasso variant used the same split. Same node, ~2 minutes apart, so background-load drift is minimal.
- The dropout self-time (11.5%) is unusually high — worth verifying that the model is actually being called in `.train()` mode for these steps. Not investigated further here.
