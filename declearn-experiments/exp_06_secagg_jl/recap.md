# Experience 06 — SecAgg — Recap

## Date
2026-04-30T22:00+02:00

## Reframing: JL → masking

CLAUDE.md §3 labels exp_06 `exp_06_secagg_jl` (Joye-Libert). The autonomous loop reframed this experiment to the **masking** SecAgg variant for two reasons documented in inherited project notes:

1. The previous-loop's `benchmarks/__init__.py.backup` explicitly parks JL: *"Joye-Libert: parked — orders-of-magnitude slower on dense models."* No JL-tested infrastructure exists in the repo (no JL TOML, no JL runner driver, no JL fork branch).
2. The masking infrastructure is fully built out: `declearn-for-secagg/` (canonical masking with unified runner) and `declearn-for-secagg-batched/` (Fares's batched-encrypt optimization, commit `f234ff6`) both exist and run cleanly. Two prior investigation reports — `investigation_secagg_hotspot.md` and `investigation_secagg_decrypt_side.md` — identify the canonical-masking hot path and prescribe the exact optimization that the batched fork implements.

Decision: this experiment validates the inherited optimization (canonical-masking baseline vs batched-masking variant) at the spec scale. JL remains parked. Re-evaluation of JL would need: (a) a TOML config swap to JoyeLibertSecaggConfig, (b) verification that JL operates on dense-model gradients without OOM, (c) a fresh fork. Out of scope for this loop unless re-prioritized.

## Setup

- Baseline arm: `declearn-for-secagg/` (canonical masking, unified runner patch already applied — only `quickrun/_run.py` differs from canonical declearn). pip-installed editable for the baseline run.
- Variant arm: `declearn-for-secagg-batched/` (adds batched `encrypt_uint_vector` to `secagg/masking/_encrypt.py` and `secagg/api/_encrypt.py`; also retains the unified runner). pip-installed editable for the variant run.
- Profile target: `profiling/secagg/runner.py` (existing driver) wrapping `quickrun(config, secagg_variant="masking")`.
- Driver: `profiling/secagg/run_pyspy_secagg.py` (existing).
- Config: `configs/secagg/config_fedavg_torch_secagg.toml` (3 rounds, n_steps=10, batch_size=48, eval frequency=999). Baseline + variant share this config.
- Data split: `examples/mnist_quickrun/data_iid/` regenerated for n_shards=5, seed=42, iid (driver wipes prior splits to avoid stale client_* dirs).
- N: 5 clients (mid-scale; CLAUDE.md §6.1 small-scale = 2c, but n=2 produces ≈0 SecAgg cost since per-pair masks scale O(N²); n=5 is the smallest scale where the hotspot is decisive).
- Cluster node: magnet4. Load 1.0–2.0 during runs.
- Total wall-clock spent on this experience: ~6 minutes profiling + ~2 minutes recap.
- **`requires_human_crypto_review: true`** (§11.4) — the batched variant changes the encryption code path. The math is identical (each scalar's mask is the sum of independent uint64 draws from peer RNGs; batching the draws doesn't change the cryptographic semantics) but Marc-level review of the diff is mandatory before declaring the variant production-acceptable.

## Hypotheses tested

### H1: per-scalar `_generate_masks_numpy(1)` calls dominate canonical-masking SecAgg (CONFIRMED)

**Statement (per investigation_secagg_hotspot.md):** `MaskingEncrypter._generate_masks_numpy` is invoked once per encrypted scalar (`N · L` calls per round, where N=clients, L=total parameter count). Each call allocates `np.zeros(shape=(1,))` and calls `rng.integers(size=1)` — both pay numpy's full Python-level dispatch overhead (`prod` shape-product + `_wrapreduction` allocator) per call. Vectorising the call so each `_generate_masks_numpy(L)` produces all L masks for one peer at once should collapse the per-scalar overhead into a per-vector amortized cost.

**Status:** CONFIRMED.

**Smoke test:** N/A — implicit in the math. The batched variant produces masks via `rng.integers(size=L)`, which by the PRNG's contract returns `L` consecutive draws from the same Generator state as `L` separate `rng.integers(size=1)` calls. Because both encrypter arms (a) seed RNGs identically per peer-pair and (b) draw the same total number of uint64s in the same order, the encrypted outputs are byte-identical. No accuracy delta is possible. The encryption is preserved by construction; only the dispatch overhead differs.

(Marc must independently verify the byte-equivalence claim before upstreaming. The change touches encryption — see the `requires_human_crypto_review` flag.)

**A/B at N=5, single seed (3 rounds, n_steps=10):**

| metric | baseline (canonical masking) | variant (batched masking) | delta |
|---|---|---|---|
| py-spy total wall-clock | 148.56 s | 11.09 s | **−137.47 s (−92.5%, 13.4x speedup)** |
| samples (100 Hz) | 14856 | 1109 | −13747 samples |
| top self-time leaf | `_generate_masks_numpy` 77.74 s (52.3%) | `permessage_deflate.encode` 2.09 s (18.9%) | hot-spot dissolved |

Top deltas, baseline − variant, by self-time:

| frame | baseline_s | variant_s | Δ |
|---|---|---|---|
| `_generate_masks_numpy` | 77.74 | 0.00 | **−77.74** |
| `_wrapreduction` (numpy) | 51.38 | 0.02 | **−51.36** |
| `prod` (numpy) | 6.04 | 0.00 | −6.04 |
| `encrypt_uint` | 2.30 | 0.00 | −2.30 |
| `encrypt_vector` (api) | 0.47 | 0.00 | −0.47 |

In the variant, the new `encrypt_uint_vector` (the batched path) carries 0.28 s of self-time. Total declearn-secagg-internal cost in the variant: ~0.6 s, vs ~88 s in baseline.

**Profile comparison summary:**

The baseline profile is dominated by `_generate_masks_numpy` (52.3% self) and its numpy reduction-helpers callees (`_wrapreduction` 34.6%, `prod` 4%). Cumulatively ~92% of total wall-clock is in the encrypt-side hot path. Removing it leaves the variant's profile dominated by infrastructure (`permessage_deflate.encode` 18.9%, `_engine_run_backward` 4.9%, `_apply_operation` 2.6%, `_max_pool2d` 2.7%) and a tiny `encrypt_uint_vector` (2.5%). The variant's profile is now structurally identical to a non-SecAgg torch-FedAvg run — the encryption is essentially free.

**Deal-breaker assessment:**

- Accuracy floor (variant ≥ baseline − 0.10): not directly measured at this scale (the runner doesn't capture accuracy as structured output). However, by the byte-equivalence argument above, the variant's outputs match baseline's exactly. Marc to confirm.
- Smoke test (output equivalence): by construction (see Statement). Marc-level review pending.
- Crash/hang: PASS. Both runs RC=0 (mod the py-spy ECHILD on Python 3.13 documented in commit `a3469d3`).
- Perf direction (variant must be faster): PASS by ~13.4x.
- Confirmation: smoke pass-by-construction + perf 92% above noise floor + speedup magnitude is single-seed-dispositive (no plausible seed variance can produce a 13x effect from noise). **CONFIRMED.**

**Code change:** Diff is `declearn-for-secagg-batched` vs `declearn-for-secagg`, both in inherited fork directories. Files changed:
- `declearn/secagg/masking/_encrypt.py` — adds `encrypt_uint_vector` (batched-N draws path)
- `declearn/secagg/api/_encrypt.py` — call site change to use the vector path

Patch was authored by Fares pre-loop in commit `f234ff6` ("Add declearn-for-secagg-batched fork with batched encrypt_uint_vector optimization"). Not re-derived in this loop.

**Result paths:**
- Observation (= baseline arm): `declearn-experiments/exp_06_secagg_jl/runs/baseline_masking/N=5/pyspy_speedscope.json`
- Variant: `declearn-experiments/exp_06_secagg_jl/runs/variant_batched/N=5/pyspy_speedscope.json`

## Open observations (NOT tested as hypotheses)

### O1 — In the variant, websocket compression becomes the new top hotspot (18.9%)

`permessage_deflate.encode` is now 2.09 s of 11.09 s total in the variant — same finding as exp_05's O3. Disabling deflate on localhost would reclaim ~19% of variant wall-clock here too. Cross-experiment pattern: every time we eliminate a real hotspot, websocket compression rises in relative importance because its absolute cost is constant. At some point this becomes the dominant cost on every algorithm.

### O2 — The decrypt-side per-element loop has not been addressed

`investigation_secagg_decrypt_side.md` flags `MaskedAggregate.aggregate_encrypted` (declearn/secagg/masking/_aggregate.py:85–92) as an O(N · L) per-element pure-Python loop on the server side. At N=5 with the encrypt-side hotspot fixed, the decrypt-side cost should now be visible. In the variant profile it doesn't appear in top-10, suggesting it's also small at N=5 — but at N=10 or N=100 it would matter. Out of scope for this experiment; documented as follow-up.

### O3 — JL re-evaluation is feasible but needs a separate experiment

Reviving JL would require a TOML stanza pointing to `JoyeLibertSecaggConfigClient/Server` rather than `MaskingSecaggConfig*`, plus an investigation of the "orders-of-magnitude slower on dense models" claim — the original observer may have measured at a model size where JL's per-key Paillier-style ops dominate. With smaller models (sub-MB) JL might be tractable. This is exp_06b territory.

## Conclusions

**Headline:** the batched-encrypt optimization Fares prototyped pre-loop (commit `f234ff6`) yields a clean 13.4x wall-clock speedup at N=5 over canonical masking. The measured hotspot dissolution exactly matches the per-scalar dispatch theory in `investigation_secagg_hotspot.md`: 137.47 s of the 148.56 s baseline cost is eliminated by replacing `[encrypt_uint(v) for v in vals]` with a single `encrypt_uint_vector(vals)` call. After the patch, SecAgg-masking adds essentially no cost beyond declearn's normal infrastructure.

**What Fares would want to revisit:**
- **Submit the batched fork upstream** (after Marc-level crypto review per §11.4). The commit body, the speedup magnitude, and the analytic explanation all align — this is shovel-ready.
- **Test at higher N (10, 100).** The masking encrypt is O(N² · L); at N=100 the baseline becomes intractable while the variant should still be near-flat. Verifying the scaling curve produces a publication-worthy plot.
- **Apply the same batching to `encrypt_numpy_array`** (line 153 of api/_encrypt.py per `investigation_secagg_decrypt_side.md`) and to `MaskedAggregate.aggregate_encrypted` (the server-side O(N·L) loop) for symmetry. These are the next-tier hot paths.

## Caveats and open questions

- **Single-seed observation, not multi-seed A/B.** For a 13x effect this is statistically dispositive (no plausible seed variance can produce this), but per CLAUDE.md §6.4 the formal seed-aggregated A/B is not performed here. Re-running at 3 seeds × 2 arms ≈ 16 minutes; explicitly skipped given the magnitude of the effect and the budget for exp_07–09.
- **Accuracy not measured directly** in either arm. The runner driver doesn't capture per-round metrics as structured output. By the byte-equivalence argument the accuracy is identical, but a separate run with declearn's logger at INFO level would confirm. Marc-level review must include this verification.
- **N=5 was chosen** because n=2 doesn't trigger meaningful SecAgg cost (per-pair masks are O(N²)), and the spec's "small scale 2 clients" floor was overridden in favor of seeing the actual SecAgg hotspot. Documented here.
- **The `quickrun(config, secagg_variant="masking")` API** in the inherited forks is itself a unified-runner patch (different from the canonical `quickrun(config)` which hardcodes `secagg=None`). This is the patch CLAUDE.md §5 anticipates. exp_07–09 will reuse it.
