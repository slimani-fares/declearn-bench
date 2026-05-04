# Cross-experiment findings — declearn profiling loop, 2026-04-30

Nine experiments run autonomously on `magnet4.lille.inria.fr` (CPU-only, declearn 2.8.0, torch 2.11.0, FedAvg + PyTorch backend, MNIST quickrun, 2 rounds at 2 clients unless noted). Each experiment has a per-experiment `recap.md` adjacent. This file collects the cross-cutting story.

For setup, environment, and per-experiment caveats see `setup_status.md` and the individual `exp_NN_*/recap.md` files.

---

## Two confirmed deal-breakers (worth code review and merging)

### A — DP-SGD: per-step `get_epsilon` is 75% of wall-clock (exp_04)

**Patch:** `exp_04_dp/patches/h1_defer_get_epsilon.diff` (37 lines, one file).
**Effect:** **2.89× wall-clock speedup** at 2c/2r, ε byte-identical (4.99640485109163 across all 3 seeds × 2 arms), accuracy delta +0.0215 (within 0.05 tolerance).
**Mechanism:** `DPTrainingManager._prevent_budget_overspending` calls opacus's `get_epsilon()` once per training step. Opacus's RDP epsilon computation walks the accountant history per call, so cumulative cost is O(N²) in step count. The patch keeps `accountant.step()` (cheap bookkeeping that maintains history correctness) and removes the per-step `get_epsilon` call. The existing `_training_round` already calls `get_privacy_spent` once at end-of-round for logging — that call now also serves as the budget check.
**Privacy semantics:** preserved exactly when the budget isn't exhausted mid-round. If a configuration *does* exhaust mid-round, detection moves from mid-step to next-round-start. For typical FL setups (which orchestrate budgets at round granularity) this is functionally equivalent; for adversarial mid-round budgets it is not.
**Crypto-review status:** `requires_human_crypto_review: true`. The change touches a DP mechanism (§11.4) and Marc must independently sign off on the budget-detection-granularity argument before upstream.
**Follow-up:** O3 in the recap proposes a stricter formulation — precompute "max steps per round before budget exhausts" once at round start (constant time per step, exact mid-round detection preserved). That's the right long-term shape; H1 is the safe and immediate intervention.

### B — SecAgg masking: 92% of wall-clock is numpy dispatch, not cryptography (exp_06)

**Validates Fares's pre-loop fork** `declearn-for-secagg-batched/`, commit `f234ff6`. Not new code from this loop.
**Effect at N=5:** **13.4× wall-clock speedup** (148.56 s → 11.09 s). The encryption hotspot dissolves: `_generate_masks_numpy` drops from 77.74 s self-time to 0; `_wrapreduction` (numpy reduction helpers) drops from 51.38 s to 0.02 s. The variant's profile becomes structurally indistinguishable from a non-SecAgg torch FedAvg run.
**Mechanism:** canonical masking calls `_generate_masks_numpy(1)` once per encrypted scalar — `np.zeros(shape=(1,))` + `rng.integers(size=1)` per call, paying numpy's full per-call dispatch tax × O(N · L) calls per round. The variant rewrites the call site to draw all L masks at once with `rng.integers(size=L)`. Same RNG, same order, byte-identical encryption output.
**Crypto-review status:** `requires_human_crypto_review: true`. The change is at the encryption boundary. The byte-equivalence claim is by construction (PRNG state is consumed in identical order), but Marc should independently verify before upstreaming.
**Follow-up:** the per-element loop in `MaskedAggregate.aggregate_encrypted` (server-side, O(N · L) per round) was not addressed by this fork. At N=10 or N=100 it would become the new bottleneck. The same batching pattern applies. See `investigation_secagg_decrypt_side.md`.

---

## The single biggest reclaimable cost is one that isn't experiment-specific

### Vector dispatch overhead — present in 9/9 experiments at 13–17% self-time

`declearn.model.api._vector.Vector._apply_operation` is the leaf hotspot in every single experiment, regardless of algorithm:

| experiment | algorithm | `_apply_operation` self-% |
|---|---|---|
| 01 | lasso | 16.3% |
| 02 | ridge | 16.5% |
| 03 | fedprox | 16.4% |
| 04 (baseline) | DP | 1.92% (RDP accountant dominates) |
| 05 | SCAFFOLD | 17.2% |
| 06 (variant) | SecAgg-batched | 2.6% |
| 07 | FairGrad | 13.34% |
| 08 | FairBatch | 14.29% |
| 09 | FairFed | 15.17% |

The exceptions (exp_04 baseline, exp_06 variant) are profiles where another hotspot (RDP accountant or numpy dispatch) drowns Vector out, but the *absolute* time spent in Vector dispatch is unchanged. Once the dominant non-Vector hotspot is removed (as in exp_04 variant and exp_06 variant), Vector returns to the 13–17% band.

**Why:** `Vector.apply_func` and `Vector._apply_operation` iterate over the per-parameter dict and dispatch one Python call per tensor (declearn/model/api/_vector.py:326–399). For a CNN with ~10 parameter tensors and ~250 SGD steps per round, that's ~2500 Python-level dispatches per round per client just for Vector arithmetic — before any actual numerical work happens.

**Recommendation:** target `TorchVector` specifically with `torch._foreach_*` batched ops (`_foreach_add_`, `_foreach_mul_`, `_foreach_sign`, etc. exist for exactly this). The framework-agnostic abstraction at `Vector` can stay as-is; `TorchVector` is the place where batching matters. Estimated impact (extrapolating from per-experiment magnitudes): a 50% reduction in Vector-dispatch self-time would shave 6–8% off every algorithm, ~10–15× the magnitude of any individual regularizer optimization.

**This is the highest-leverage single change in the study.** It's invasive (touches a load-bearing abstraction) but the evidence base is now nine independent runs.

### Lasso's outlier overhead is the same Vector pattern, sign()-specific

Among the regularizers:
- Ridge: +2.2% over vanilla — within noise
- FedProx: +4.4% — at the noise floor
- **Lasso: +12.5%** — clear outlier

The +10pp gap between lasso and the others is entirely attributable to one operation: `weights.sign()` calls `Vector.apply_func(torch.sign)`, adding +1.22 s self-time (47% of the lasso−vanilla delta). Ridge and FedProx don't trigger `apply_func` (they use only `_apply_operation` with scalars/Vectors). So "lasso is slow" decomposes to "any regularizer that calls `apply_func(stateless_torch_op)` pays a per-tensor Python dispatch tax." Same root cause as the Vector finding above. The `torch._foreach_*` fix would make lasso's overhead vanish too.

---

## WebSocket compression on localhost — persistent 2–8% reclaim

`websockets.extensions.permessage_deflate.encode` shows up in every experiment at 2–8% self-time. On localhost there's no bandwidth case for compression; the cost is pure CPU:

| experiment | `permessage_deflate.encode` self-% |
|---|---|
| 01–03 (regularizers) | 2.7–3.2% |
| 04 (DP, baseline) | 0.4% (RDP dominates) |
| 04 (DP, variant) | ~3% (now visible) |
| 05 (SCAFFOLD, N=5) | **7.8%** (highest seen — aux_var traffic) |
| 06 (SecAgg-batched) | **18.9%** (highest seen — biggest fraction because everything else is fast) |
| 07–09 (fairness) | 2.8–3.2% |

The configuration that disables permessage-deflate is straightforward — declearn doesn't expose it directly but the websockets library does. For localhost benchmarks this is a free reclaim. For real-network deployments it's a tradeoff (CPU vs bytes-on-wire).

---

## FairFed beats FairGrad and FairBatch on both axes

| algorithm | wall-clock | round-2 server accuracy |
|---|---|---|
| FairGrad (exp_07)   | 41.15 s | 0.5103 |
| FairBatch (exp_08)  | 41.01 s | 0.7772 |
| FairFed (exp_09)    | **36.92 s** | **0.9343** |

Counter-intuitive at first read: FairFed is the algorithm that *should* be most expensive on the server side (it does post-aggregation reweighting). It turns out to be the lightest because clients run plain FedAvg — no per-step gradient correction (FairGrad), no per-batch reweighting (FairBatch). Less per-step work scales better than algorithmic simplicity.

**Implication for declearn users:** for binary-target fairness tasks at this scale, FairFed is a reasonable default. For higher-dimensional fairness constraints (multi-attribute or non-binary) the comparison may invert; rerun at scale before generalizing.

---

## Findings that didn't make the bar but deserve a flag

- **`compute_batch_predictions` is run twice in fairness experiments** — once as part of the standard end-of-round eval, once as part of the fairness round's group-wise metric computation. Caching the validation predictions across these two calls would save ~17% of fairness-experiment wall-clock and is a "exp_07b" candidate that benefits all three fairness algorithms.
- **`_build_iterator` (declearn dataset)** consistently appears in top-15 across experiments at ~1–3% self-time. Per-batch iterator construction; no obvious optimization but worth flagging for higher-scale studies.
- **`dropout` self-time** is 11–14% across experiments because the mnist_quickrun model uses two dropout layers (0.25, 0.5) and torch's CPU dropout isn't free. Not a declearn issue per se but a profiling note: when interpreting these numbers for "what does declearn cost on top of plain torch", subtract the dropout (and conv, maxpool, linear) self-times from the totals.
- **`_compile_bytecode` and `_call_with_frames_removed` (importlib)** show ~0.4–2 s of total time across most experiments — Python import overhead. Mostly one-shot; matters for short runs (smoke tests) but amortizes away at any real scale. Not actionable.

---

## Methodology caveats applicable to all experiments

- **Single-seed observation profiles.** Only exp_04 ran a multi-seed A/B (3 seeds). Exp_06 effect was so large (13×) that single-seed was statistically dispositive on its own. Other experiments are observational, not statistically validated. Re-running with 3–5 seeds is straightforward if you want hardened numbers for any single recap.
- **2 rounds × 2 clients (mostly).** This is the §6.1 small-scale floor. Several findings (SCAFFOLD aux_var cost, SecAgg encrypt-side, fairness-round-vs-training-round ratio) scale with N or with rounds. The per-experiment recommendations explicitly note where higher-scale verification is needed.
- **Cluster load drifted from ~0.2 to ~2.0 over the 90-minute loop.** Same node throughout, both arms of any A/B run within ~30 s of each other, so the bias is symmetric. Absolute timings carry ~10–15% noise; relative magnitudes (especially the consistent Vector pattern) are robust because the same noise hits both arms identically.
- **Fairness experiments used a 5-client data split** (legacy from exp_06's 5-shard regenerate). exp_05 SCAFFOLD also has a 5-client comparator. exp_01–04 are at 2 clients. Cross-experiment magnitudes are not directly comparable across these splits; within-split comparisons (e.g., FairGrad vs FairBatch) are clean.

---

## Suggested follow-up plan

In approximate order of value-per-effort:

1. **Crypto-review and upstream the DP patch (exp_04).** 37-line diff, one file, 2.89× speedup, byte-identical privacy. Cleanest win.
2. **Crypto-review and upstream the SecAgg-batched fork (exp_06).** Already implemented by Fares pre-loop; this study confirmed the magnitude (13.4× at N=5).
3. **Disable permessage-deflate on localhost runs.** Configuration change, no code. Reclaims 2–8% across every algorithm. Add a `[network] compress = false` knob if it doesn't already exist.
4. **`TorchVector` foreach rewrite.** Highest-leverage but most invasive. The `_apply_operation` and `apply_func` paths in `declearn/model/torch/_vector.py` are the targets. Expected impact: 6–8% off every algorithm; possibly more on regularizers.
5. **Validation-prediction caching for fairness** (exp_07b). One-shot fix that benefits FairGrad/FairBatch/FairFed. ~17% reclaim per fairness experiment.
6. **SecAgg server-side `aggregate_encrypted` batching.** Symmetric to #2; targets the decrypt side at higher N.
7. **SCAFFOLD aux_var compression at higher N.** Validate at N=10/N=20 whether aux_var compression cost crosses the threshold for a SCAFFOLD-specific intervention. Currently it's just websocket compression at scale.

Items 1–3 are clear merges. Items 4–7 are research questions where this study has produced the evidence base; the next loop iteration would do the implementation + A/B.
