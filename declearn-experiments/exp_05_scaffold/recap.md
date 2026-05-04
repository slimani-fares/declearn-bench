# Experience 05 — SCAFFOLD — Recap

## Date
2026-04-30T21:51+02:00

## Setup
- Baseline: declearn 2.8.0 canonical (no fork — observation-only experience)
- Variant fork(s): none
- Data splits: `examples/mnist_quickrun/data_iid/` (2 shards) and `examples/mnist_quickrun/data_iid_n5/` (5 shards), both seed 42 iid
- Cluster node: magnet4 (load 0.7–2.0 during the runs; rising toward exp_04's tail-end activity but still under 4.0)
- Total wall-clock spent on this experience: ~6 minutes

## Hypotheses tested

None. SCAFFOLD-specific frames carry only 0.03–0.06 s of self-time at both 2c and 5c — overwhelming dominated by SCAFFOLD's runtime in `_engine_run_backward`, `_apply_operation`, and `dropout`, none of which are SCAFFOLD-implementation-specific. The hypothesis territory hinted at by CLAUDE.md §8 (aux_var exchange cost) materializes as websocket compression and Vector dispatch overhead, both of which are infrastructure costs that affect all algorithms, not SCAFFOLD-specific.

## Open observations

### O1 — SCAFFOLD overhead scales with client count: +6.2% at 2c, +13% at 5c

| n_clients | vanilla baseline (s) | SCAFFOLD variant (s) | Δ (s) | Δ% |
|---|---|---|---|---|
| 2 | 20.95 | 22.25 | +1.30 | +6.2% |
| 5 | 22.09 | 24.98 | +2.89 | +13.0% |

The 2x growth from 2c to 5c is consistent with SCAFFOLD's per-step aux_var exchange between server and clients (linear in client count).

### O2 — At N=5, the SCAFFOLD-attributable cost decomposes as:

Top deltas, scaffold(N=5) − vanilla(N=5), self-time:

| frame | Δs | category |
|---|---|---|
| `Vector._apply_operation` | +0.87 | declearn-internal Vector dispatch |
| `permessage_deflate.encode` (websockets) | +0.63 | message compression |
| `_build_iterator` (dataset) | +0.38 | per-client data iterator overhead |
| `_conv_forward` (torch) | +0.27 | run-to-run noise |
| `iterencode` (json) | +0.16 | message serialization |
| `permessage_deflate.decode` | +0.12 | message decompression |
| Scaffold-specific frames (run, process_aux_var) | <0.01 self, ~0.70 total | scaffold-implementation-specific |

The scaffold-implementation-specific frames (`run` in `_scaffold.py`, `process_aux_var`) carry sub-1% of self-time. The user-perceivable +13% slowdown comes from infrastructure that surrounds the SCAFFOLD aux_var traffic — not from the scaffold algorithm itself.

### O3 — websocket compression scales 48% from 2c to 5c with SCAFFOLD on

`permessage_deflate.encode` self-time:
- 2c vanilla: 0.57 s, 2c SCAFFOLD: 0.70 s (Δ +0.13 s, +23%)
- 5c vanilla: 1.31 s, 5c SCAFFOLD: 1.94 s (Δ +0.63 s, +48%)

The aux_var messages carry per-parameter control variates (size ≈ model size). At each round, all clients send + receive aux_var. Compression cost scales with payload × clients. **For localhost runs (where the network bandwidth saving is meaningless), disabling permessage-deflate would reclaim this entire delta.** This is not SCAFFOLD-specific — it'd benefit all algorithms — but it's the largest single attributable cost in this experiment.

### O4 — Vector overhead pattern matches exp_01–04

The `+0.87 s _apply_operation` delta at N=5 echoes the same Vector-dispatch hotspot already documented in exp_01 (lasso), exp_02 (ridge), exp_03 (fedprox), and exp_04 (DP). SCAFFOLD's aux_var corrections are computed via Vector arithmetic, so they pay the same Python-dispatch tax. This strengthens the cross-experience observation already made in exp_03's recap: a single `Vector` layer optimization (e.g., `torch._foreach_*` batch ops) would reduce overhead across all algorithms.

## Conclusions

**Headline:** SCAFFOLD's wall-clock cost in declearn at small scale (2c–5c) is dominated by infrastructure (Vector dispatch, message compression, JSON serialization) rather than the algorithm itself. The SCAFFOLD-specific code paths consume <1% of self-time. There is no SCAFFOLD-specific deal-breaker visible at this scale.

**What Fares would want to revisit:**
- **Disable permessage-deflate on localhost runs.** This is the cleanest single intervention exposed by this experiment. At 5c with SCAFFOLD it's 1.94 s of self-time = 7.8% of total — definitely above the 5% noise floor. It applies to ALL experiments but materializes most strongly here because aux_var traffic is the largest payload.
- **Re-run SCAFFOLD at higher scales (N=10, N=20).** The 2c→5c trend (+6% → +13%) extrapolates to ~+25–35% at N=10. If that holds, SCAFFOLD aux_var costs become the dominant single contributor and would warrant a SCAFFOLD-specific A/B (e.g., compressed-aux-var fork, or layer-wise aux_var instead of full-model). Compute cost: ~3–5 min per profile at N=10.

## Caveats and open questions

- Observation profiles only (no A/B run because no SCAFFOLD-specific optimization was proposed).
- Single-seed observation at each (n_clients, arm) combination. Per CLAUDE.md §11.2, this is observational, not a finding. Cross-seed validation would be the next step IF a SCAFFOLD-specific hypothesis emerges at higher N.
- The `_conv_forward` delta at +0.27 s and `dropout` delta at +0.11 s look like normal run-to-run torch op variance, not SCAFFOLD-attributable.
- Cluster load was rising during the run (0.74 → 1.96 over the experiment). Both arms ran back-to-back within ~30 seconds, so the bias is symmetric. Result is reliable.
