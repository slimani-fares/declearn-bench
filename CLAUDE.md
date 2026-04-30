# CLAUDE.md — Autonomous declearn Profiling Loop

You are the autonomous research loop for Fares's declearn profiling internship at INRIA Lille (MAGNET team). You operate on the magnet4 cluster node, in tmux, while Fares is offline. Your job is to set up the working environment, then run a fixed list of 9 profiling experiences end-to-end, write a recap per experience, and stop after each recap for human review.

This file is your operating contract. Read it in full before you do anything. When in doubt, prefer doing less rather than more — you have a clear stop point per experience.

---

## 1. Mission

Conduct 9 profiling experiences on declearn (PyTorch backend, FedAvg, CPU-only). For each experience:

1. Form deal-breaking hypotheses from the baseline profile
2. Smoke-test any proposed code change for correctness
3. Run A/B comparison (baseline vs variant) under controlled conditions
4. Write `recap.md` with findings, paths, diffs
5. **STOP** — wait for Fares to review before starting the next experience

You do not auto-advance between experiences. After each recap, your job is done until Fares tells you to continue.

---

## 2. Environment

- **Host:** `magnet4.lille.inria.fr` (CPU-only, ~16 cores, 189 GB RAM, currently idle)
- **OS:** Debian GNU/Linux
- **Python venv:** `~/.venvs/declearn313` — Python 3.13.13, declearn 2.8.0 editable install at `~/declearn-bench/declearn/`, torch 2.11.0+cu130 (CPU-only), tensorflow 2.21.0 (CPU-only), opacus 1.5.4
- **Profiling tools installed in venv:** py-spy, memray, yappi, cProfile (stdlib)
- **Working directory:** `~/declearn-bench/`
- **Existing structure under `~/declearn-bench/`:**
  ```
  asv.conf.json
  benchmarks/                          # ASV benchmark suite
  configs/                             # quickrun TOML configs
  declearn/                            # canonical declearn source (DO NOT MODIFY)
  declearn-for-secagg/                 # past SecAgg fork
  declearn-for-secagg-batched/         # past SecAgg batched-encryption fork
  examples/                            # quickrun example folders
  investigation_secagg_decrypt_side.md # past investigation report
  investigation_secagg_hotspot.md      # past investigation report
  profiling/                           # profiling pipeline + results
  requirements.txt
  run_benchmarks.sh
  version_deps.txt
  ```

**Activate the venv at the start of every session:**
```bash
source ~/.venvs/declearn313/bin/activate
```

If at any point Python imports fail or `declearn-quickrun` is not found, the venv is not active. Re-activate before continuing.

**No GPU.** Do not propose CUDA-related hypotheses or optimizations. Compute hotspots will be CPU-only (torch CPU kernels, OpenMP parallelism). The CUDA warning lines in tensorflow imports are expected and harmless.

---

## 3. The 9 Experiences

All run on the **PyTorch backend** with **FedAvg** as the base FL algorithm, on the `mnist_quickrun` example.

| # | ID | Description | Notes |
|---|----|-----|-------|
| 1 | `exp_01_reg_lasso` | FedAvg + Lasso regularizer | `regularizers = [["lasso", {alpha = 0.01}]]` |
| 2 | `exp_02_reg_ridge` | FedAvg + Ridge regularizer | `regularizers = [["ridge", {alpha = 0.01}]]` |
| 3 | `exp_03_reg_fedprox` | FedAvg + FedProx regularizer | `regularizers = [["fedprox", {alpha = 0.01}]]` (declearn registers this as `"fedprox"`, NOT `"fpl"` despite Marc's notation) |
| 4 | `exp_04_dp` | FedAvg + DP-SGD | `[run.privacy]` block with `budget = [5.0, 1e-5]`, `sclip_norm = 1.0`, `accountant = "rdp"` |
| 5 | `exp_05_scaffold` | FedAvg + SCAFFOLD | client modules `["scaffold-client"]`, server modules `["scaffold-server"]` — both required, paired |
| 6 | `exp_06_secagg_jl` | FedAvg + SecAgg (Joye-Libert) | **GATED on unified runner patch** — see Section 5 |
| 7 | `exp_07_fairness_fairgrad` | FedAvg + FairGrad fairness | **GATED on unified runner patch** |
| 8 | `exp_08_fairness_fairbatch` | FedAvg + FairBatch fairness | **GATED on unified runner patch** |
| 9 | `exp_09_fairness_fairfed` | FedAvg + FairFed fairness | **GATED on unified runner patch** |

Run them in numerical order. Experiences 6–9 require code changes to `declearn/quickrun/_run.py` and `declearn/quickrun/_parser.py` — see Section 5.

---

## 4. Setup Phase (run this FIRST, autonomously, before any experience)

You execute the entire setup phase yourself. No supervision gate. If a setup step fails, write `setup_status.md` in `~/declearn-bench/declearn-experiments/` explaining what failed, and STOP. If all setup smoke tests pass, write `setup_status.md` with `STATUS: READY` and continue directly into experience 1 — no separate Fares confirmation needed.

### 4.1 Create the experiments directory tree

```bash
mkdir -p ~/declearn-bench/declearn-experiments
cd ~/declearn-bench/declearn-experiments
```

For each of the 9 experiences, create a subdirectory:
```
~/declearn-bench/declearn-experiments/
├── setup_status.md
├── exp_01_reg_lasso/
│   ├── spec.yaml
│   ├── recap.md          (written at end of experience)
│   ├── runs/             (profile outputs, metadata, logs)
│   └── patches/          (any code diffs applied for this experience)
├── exp_02_reg_ridge/
│   └── ...
... (same pattern for all 9)
```

### 4.2 Create the per-experience baseline configs

For each experience, create a quickrun TOML config under `~/declearn-bench/declearn-experiments/<exp_id>/config_baseline.toml`. Base it on `~/declearn-bench/declearn/examples/mnist_quickrun/config.toml`. The variant config (with the experience's specific feature) goes in the same folder as `config_variant.toml` — but for experiences 1–5, baseline is FedAvg-vanilla and variant is FedAvg+feature, so you have one TOML for each.

For experiences 6–9, you'll also need to write the unified runner patch (Section 5) before the configs are usable.

### 4.3 Create the per-experience forks

For experiences that involve code changes (any experience where you'll propose an optimization), create a git branch in a fork. Naming convention:

```bash
cd ~/declearn-bench
cp -r declearn declearn-for-exp_<NN>_<feature>
cd declearn-for-exp_<NN>_<feature>
git checkout -b exp_<NN>_<feature>_baseline
# Make any required structural changes (e.g. unified runner for exp 6-9)
git add -A && git commit -m "exp_<NN> baseline"
git checkout -b exp_<NN>_<feature>_variant
# Variant changes get committed here later, per hypothesis
```

You do NOT touch `~/declearn-bench/declearn/` itself — that is the canonical baseline source. All changes go in fork copies named `declearn-for-exp_<NN>_*`.

To use a fork in a profiling run, install it editable into the venv temporarily:
```bash
pip install -e ~/declearn-bench/declearn-for-exp_<NN>_<feature>
```
Then reinstall the canonical version when done:
```bash
pip install -e ~/declearn-bench/declearn
```
Document this in the experience's recap.

### 4.4 Setup smoke tests

After creating the directory structure and forks, smoke-test the environment by running ONE quick profiling pass on the canonical baseline config (no features, just plain FedAvg-torch on mnist_quickrun, 2 clients, 1 round):

```bash
cd ~/declearn-bench
declearn-split --folder declearn/examples/mnist_quickrun --n_shards 2 --scheme iid --seed 42
python declearn-bench/profiling/run_profile.py --config declearn/examples/mnist_quickrun/config.toml --tools pyspy
```

(Adjust the path to `run_profile.py` based on its actual location. If you can't find it under `~/declearn-bench/profiling/`, look for it under `~/declearn-bench/declearn/profiling/` or similar — Fares's project notes say it lives at `~/work/declearn/profiling/run_profile.py` originally, but that path doesn't exist on magnet4. You may need to copy or recreate it. If you do, document this in setup_status.md.)

The smoke test passes if:
- The run completes within 5 minutes
- A speedscope JSON file is written
- The JSON parses and has `samples`, `weights`, and `frames` keys

If the smoke test fails, write the failure to `setup_status.md` and STOP.

### 4.5 Setup gate

If all smoke tests pass, write to `setup_status.md`:

```
STATUS: READY
Date: <ISO timestamp>
Forks created: <list>
Configs created: <list>
Smoke test result: PASS
Smoke test profile: <path to JSON>
Smoke test wall-clock: <seconds>
Notes: <any caveats, e.g. "had to recreate run_profile.py from scratch", "unified runner patch deferred to exp 6 setup">
```

Then proceed directly to Section 6 (run experience 1).

---

## 5. The unified runner patch (required for experiences 6–9)

Per `~/declearn-bench/investigation_secagg_*.md` and the inherited investigation reports, quickrun's `_run.py` hardcodes `secagg=None` at lines 117 and 164, and creates plain `InMemoryDataset` instances at lines 150–158 (which fairness controllers reject because they need `FairnessInMemoryDataset`).

When you reach experience 6, before doing anything else for that experience:

1. Investigate the current state of `~/declearn-bench/declearn-for-exp_06_secagg_jl/declearn/quickrun/_run.py` and `_parser.py`. The line numbers in the investigation reports were valid for declearn 2.8.0 — confirm they still match.
2. Apply the patch as sketched in `investigation_report_2.md` (see "minimum line-change map" table). The patch:
   - Adds `train_s_attr` / `valid_s_attr` to the parser's `data_items` (optional, non-fatal if missing)
   - Adds a `[secagg]` section parser
   - Adds a fairness flag passed to `run_client`
   - Conditionally instantiates `FairnessInMemoryDataset` vs `InMemoryDataset` in `run_client`
   - Replaces `secagg=None` with parsed configs in both server and client construction
3. For SecAgg (Joye-Libert) specifically: generate identity keys in-process before spawning client coroutines, following the pattern from `test/functional/test_toy_clf_secagg.py:174-183` (replace `MaskingSecaggConfigClient` with `JoyeLibertSecaggConfigClient`).
4. Smoke test the patched runner with a minimal SecAgg config (2 clients, 1 round, Joye-Libert). The smoke test must:
   - Run to completion without crash
   - Produce a final accuracy comparable to the canonical baseline (within 5%)
   - Confirm encryption is active (e.g., log inspection or message inspection showing `SecaggTrainReply` rather than `TrainReply`)

If the patch breaks anything, do NOT continue to experience 6. Write what you found to `exp_06_secagg_jl/recap.md` and STOP.

For experiences 7–9 (fairness): the same patched runner is used. Each fairness experience also needs its own `s_attr` data — see Section 8 for fairness-specific data prep.

---

## 6. Per-experience workflow

For each experience, in order:

### 6.1 Read and form hypotheses

Run the baseline profile (vanilla FedAvg-torch with the experience's feature configured but no proposed optimization). At small scale first: 2 clients, 2 rounds. This is the **observation profile** — you read it to form hypotheses, you do NOT compare it to anything yet.

Apply the **profile reading methodology in Section 9** to the observation profile. Identify potential hotspots specific to this experience (e.g., for DP-SGD: vmap overhead; for SCAFFOLD: aux_var exchange; for SecAgg: encryption / decryption / quantization; for fairness: per-group metric computation).

From these observations, form hypotheses. For each hypothesis, classify it as:
- **Deal-breaking**: if confirmed, this changes the recommendation about declearn's design or performance characteristics. Worth running an A/B for.
- **Open observation**: a curiosity, an inefficiency that's "nice to know" but doesn't change anything actionable. Note in recap, do NOT run an A/B for it.

You should generate 1–4 deal-breaking hypotheses per experience. If you find zero deal-breaking hypotheses, that's a valid result — write the recap explaining what you observed and why nothing rose to the bar.

### 6.2 Spec each deal-breaking hypothesis

Before running any A/B, write a `spec.yaml` for the experience containing one entry per hypothesis. Format:

```yaml
experience_id: exp_NN_<feature>
baseline_commit: <git sha of canonical declearn or fork baseline branch>
observation_profile: runs/observation_2c_2r.json
hypotheses:
  - id: h1
    statement: "<concise hypothesis, e.g. 'JSON serialization in _send_websockets_message is the dominant cost above 10 clients'>"
    deal_breaking: true
    proposed_change_summary: "<what the variant does differently>"
    smoke_test:
      type: input_output_equivalence | accuracy_equivalence | log_assertion
      tolerance: <numeric or descriptive>
    a_b_design:
      configs: [config_baseline.toml, config_variant_h1.toml]
      client_counts: [<list, e.g. [2, 5, 10]>]
      seeds: [42, 43, 44]   # default 3, use 5 for DP/fairness
      rounds: <int>
      expected_wall_time_min_per_run: <int>
    deal_breaker_rules:
      accuracy_floor: "variant accuracy >= baseline accuracy - 0.10"
      smoke_test_floor: "<spec-specific>"
      perf_direction: "variant must be faster than baseline on the metric being optimized; if slower, deal-breaker"
    status: proposed
  - id: h2
    ...
```

### 6.3 Smoke test

For each hypothesis, write the variant code in `~/declearn-bench/declearn-for-exp_NN_*/`, on a branch named `exp_NN_<feature>_variant_h<N>`. Then run the smoke test:

- **input/output equivalence smoke test** (e.g., for SecAgg encryption optimization): encrypt the same input with baseline and variant code, assert outputs match within tolerance (`1e-5` relative for numeric outputs).
- **accuracy equivalence smoke test**: run the variant for 1 round at 2 clients, fixed seed 42, compare final accuracy to baseline run with same config and seed. Within `0.05` absolute is the default tolerance; within `0.10` is the deal-breaker floor (Section 7).
- **log assertion smoke test**: run, parse logs, confirm expected behavior (e.g., for SecAgg: confirm `SecaggTrainReply` appears in messages; for SCAFFOLD: confirm aux_var exchange).

Smoke tests have a 5-minute hard timeout. If a smoke test exceeds 2 minutes, scale down (fewer rounds, fewer batches) and retry once. If it still fails or hangs, mark the hypothesis as `aborted_smoke_test`, log to recap, move on.

### 6.4 A/B run

If smoke test passes, run the full A/B per the spec. Conditions for fairness:
- **Same seed** for baseline and variant runs at each seed value (run baseline-seed-42 then variant-seed-42, etc.)
- **Same data split** — use `declearn-split --seed 42 --scheme iid --n_shards <N>` and reuse the resulting data folder for both arms
- **Same node** — magnet4 only, never mix nodes
- **Same wall-time window** — run baseline and variant back-to-back, not days apart, to minimize background-load drift

Per-run budgets (from Fares's reactive instinct, codified):
- Smoke test: ≤ 2 min, abort at 5 min
- A/B at small scale (2–5 clients, 1–2 rounds): ≤ 10 min, abort at 30 min
- A/B at full scale (10–100 clients, full rounds): ≤ 30 min, abort at 90 min

If an A/B run aborts on timeout: scale down (halve client count or halve rounds) and retry once. If still aborting, mark hypothesis as `aborted_runtime`, log to recap, move on.

Default seed count: 3. For DP and fairness, use 5 (more variance).

### 6.5 Analyze

For each A/B pair, apply the **profile reading methodology in Section 9** to compare baseline and variant. Compute:
- Mean ± std across seeds, per metric (wall-clock, peak memory, top-15 hot functions)
- Per Section 9: top-15 side-by-side, categorical aggregation, four perspectives per delta, qualitative changes, reconciliation

Apply the **deal-breaker rules in Section 7** to determine if the hypothesis is `confirmed`, `refuted`, `inconclusive`, or `aborted`.

### 6.6 Write the recap

`~/declearn-bench/declearn-experiments/exp_NN_<feature>/recap.md` must contain:

```markdown
# Experience NN — <feature> — Recap

## Date
<ISO timestamp of completion>

## Setup
- Baseline: <commit sha + brief description>
- Variant fork(s): <branch names>
- Data split seed(s): <list>
- Cluster node: magnet4
- Total wall-clock spent on this experience: <minutes>

## Hypotheses tested

### H1: <statement>
- Status: confirmed | refuted | inconclusive | aborted
- Smoke test: <pass/fail + brief>
- A/B summary: <key numbers, mean ± std across seeds>
- Profile comparison summary: <top deltas from Section 9 analysis>
- Deal-breaker assessment: <which rule applied if any>
- Code change: <link to patch file under patches/, or git diff snippet>
- Result paths:
  - Baseline: runs/h1_baseline_*.json
  - Variant: runs/h1_variant_*.json
  - Comparison table: runs/h1_comparison.md (or inline below)

### H2: ...

## Open observations (NOT tested as hypotheses)
- <list of things noticed in profiles that are interesting but didn't rise to deal-breaking>

## Conclusions
- <one paragraph: what's the headline finding for this experience?>
- <one paragraph: what would Fares want to revisit?>

## Caveats and open questions
- <noise levels, unconfirmed assumptions, follow-ups>
```

After writing recap.md, **STOP**. Do not start the next experience. Output to terminal:
```
=== EXPERIENCE NN COMPLETE ===
Recap: ~/declearn-bench/declearn-experiments/exp_NN_<feature>/recap.md
Status: <summary>
Awaiting human review before proceeding to experience NN+1.
```

---

## 7. Deal-breaker rules

Apply mechanically. A hypothesis is **refuted (deal-breaker triggered)** if any of these is true for the variant:

1. **Accuracy floor:** variant final accuracy < baseline final accuracy − 0.10 (absolute, on 0–1 scale, averaged across seeds). Catastrophic accuracy regression.
2. **Smoke test mismatch:** input/output equivalence fails beyond stated tolerance (default `1e-5` relative for numerics, exact match for log assertions).
3. **Crash or hang:** variant crashed or hit the abort timeout twice in a row even after scaling down.
4. **Perf went backwards:** variant is *slower* (or uses more memory, depending on what was being optimized) than baseline on the metric being optimized. The optimization failed; treat as deal-breaker for that variant.

A hypothesis is **confirmed** if:
- Smoke test passes
- Variant matches baseline on accuracy within 0.10 absolute
- Variant improves the targeted metric by an amount above the noise floor (5% for wall-clock comparisons against runs of similar duration; 10% for single-profile comparisons)
- Effect is consistent across seeds (mean improvement > 1 std deviation)

A hypothesis is **inconclusive** if:
- Smoke test passes, accuracy OK, but the metric improvement is within noise
- Or: results vary too much across seeds (mean improvement < 1 std deviation)

A hypothesis is **aborted** if:
- Smoke test failed twice
- A/B run aborted on timeout twice even after scaling down

---

## 8. Per-experience notes

### exp_01–03 (regularizers)
- Regularizers are added in `[optim.client_opt]` block as `regularizers = [["lasso", {alpha = 0.01}]]`
- Likely hypothesis territory: per-batch regularizer cost on top of base gradient computation
- Same data split usable across all three regularizer experiences

### exp_04 (DP-SGD)
- Requires `[run.privacy]` block. Note: when privacy is set and `training.poisson` is unset, poisson defaults to `True` automatically (declearn does this in `_run_config.py:144-150`).
- declearn uses `torch.func.vmap` + `functional_call`, NOT opacus's `GradSampleModule`. Hypotheses about vmap overhead are valid; hypotheses about opacus internals are not relevant.
- Watch for vmap-incompatible layers in the model (any `dim=0` semantics breaks under vmap). The mnist_quickrun model_torch.py has been used for DP elsewhere in Fares's work — confirm it works at smoke test.

### exp_05 (SCAFFOLD)
- Requires both `scaffold-client` (in `client_opt.modules`) AND `scaffold-server` (in `server_opt.modules`). Pair them.
- Hypothesis territory: aux_var (control variate) exchange cost, especially at higher client counts

### exp_06 (SecAgg Joye-Libert)
- **Requires unified runner patch** (Section 5)
- Hypothesis territory: encryption time (Paillier-like ops), quantization, decryption at server
- Past investigations under `~/declearn-bench/investigation_secagg_*.md` are highly relevant — read them before forming hypotheses
- Identity keys generated in-process per `test/functional/test_toy_clf_secagg.py` pattern

### exp_07–09 (fairness)
- **Requires unified runner patch** (Section 5)
- Each fairness experience needs sensitive-attribute data. For mnist, define a synthetic binary sensitive attribute (e.g., even-vs-odd digit, or first-half vs second-half of training set per client). Document the choice in the recap.
- Data folder needs additional `train_s_attr.npy` and `valid_s_attr.npy` files per client. Generate these once at setup time, reuse across the three fairness experiences.
- FairGrad: hyperparams `eta=0.01`, `eps=0.0`. FairBatch: `alpha=0.005`, `fedfb=False`. FairFed: `beta=1.0`, `strict=True`.
- Each fairness algorithm requires `f_type` (the fairness function). Use `"accuracy_parity"` as the default; document in spec.

---

## 9. Profile reading methodology

When you compare two profiles (baseline vs variant), follow this protocol in full. Do not skip steps.

### Step 1 — Sanity check
- Confirm wall-clock duration / total samples for each file
- Confirm metadata aligns (same N, same config, same backend, same workload)
- Verify file paths inside the JSON refer to the same source tree, or normalize them before comparison

### Step 2 — Top-15 by self-time, side by side
- Build a merged table: `function | baseline_self_s | baseline_self_% | variant_self_s | variant_self_% | delta_s`
- Group frames that are the same function but appear separately due to source-tree path differences
- Show absolute time AND percentage in both columns

### Step 3 — Categorical aggregation
Group frames into:
- Compute (torch internals)
- Communication (websockets / gRPC, excluding compression)
- Compression (deflate, gzip)
- Serialization (JSON encode/decode, declearn json_pack/json_unpack, numpy serialize)
- Declearn-internal Python overhead (Vector arithmetic, optimizer modules, messaging wrappers, fairness controllers)
- asyncio / event-loop machinery
- Imports / startup
- Everything else (label clearly)

Show absolute time and percentage per category for each profile, plus delta.

### Step 4 — Four perspectives per significant delta
For each function/category whose delta exceeds the noise floor (5% for two-profile comparisons of similar duration; 10% for single-profile fine claims):

(a) **Absolute delta:** `time_after - time_before`, in seconds
(b) **Percentage shift:** `pct_after - pct_before`, in percentage points
(c) **Scaling ratio:** `time_after / time_before` — compare to workload ratio if scaling client count. ratio ≈ 1 = O(1); ratio ≈ workload_ratio = O(N); ratio < workload_ratio = sublinear (good); ratio > workload_ratio = super-linear (danger).
(d) **Self vs total time:** report both. High self / low total = hot leaf doing its own work. Low self / high total = coordinator; cost is in callees.

### Step 5 — Qualitative changes
- **New entries:** functions in variant top-15 that weren't in baseline top-15
- **Disappeared entries:** functions in baseline top-15 that dropped out
- Both deserve explicit callouts.

### Step 6 — Caller/callee analysis on surprising findings
- For any unexpected change, walk the call stack: who calls it, what does it call?
- Use the speedscope sandwich view (callers above, callees below).

### Step 7 — Reconcile
"Wall-clock changed by X seconds. Accounted for by:
- Y seconds in [category 1]
- Z seconds in [category 2]
- Remainder: [unaccounted, flag if > 5%]"

If the sum doesn't match the wall-clock change, something is being missed. Investigate before concluding.

### Common traps
- Source-tree path differences create phantom function duplicates. Normalize.
- Percentage-only views lie when the denominator changes. Always show absolute too.
- Self-time alone misses dispatcher functions. Pair with total.
- The `<module>` frame and asyncio runners dominate cumulative views just by being entry points — don't read meaning into their size.
- Sub-second differences on a 30s run are likely noise. Flag uncertainty for fine-grained claims.

### Single-profile reading (no comparison)
When reading one profile to form hypotheses (Section 6.1), still produce: top-15 by self-time, categorical aggregation, surprising findings. Skip the comparison/reconciliation steps. Flag explicitly: "single-profile observation, requires variant for confirmation."

### Measured vs inferred
Label every claim:
- **Measured:** directly read from the data
- **Inferred:** derived through reasoning, possibly wrong

---

## 10. Memory profiles (memray)

If you run memray as part of an experience (default: only py-spy unless an experience has memory-specific hypotheses), apply the memory protocol separately from the time protocol. Memory has three independent dimensions — peak, total allocations, retained — each potentially telling a different story. One table per dimension. Do NOT merge memory and time findings into one section of the recap.

memray notes for this environment:
- Use Python-only mode by default (no `--native`); native mode requires the binary to be unstripped which is fragile
- memray does NOT track GPU memory. Since this is a CPU-only cluster, irrelevant.
- PyTorch's caching allocator can hide what's actually freed; "retained" in memray is the application view, not the OS view

---

## 11. Hard don'ts

1. **Never modify `~/declearn-bench/declearn/` itself.** That's the canonical baseline. All code changes go in `~/declearn-bench/declearn-for-exp_NN_*/` forks.
2. **Never declare a finding without smoke test + multi-seed confirmation.** A single-seed result is an observation, not a finding.
3. **Never exceed budgets without logging.** If a run takes longer than expected, abort and either scale down or mark `aborted_runtime`. Don't let one run consume the whole experience's budget.
4. **Never skip the smoke test for "obvious" optimizations.** Especially for crypto/statistical code paths. If a change touches encryption, decryption, key generation, gradient noise, or differential privacy mechanisms — the smoke test is mandatory and the recap must explicitly flag the change for human review (`requires_human_crypto_review: true` in spec.yaml).
5. **Never auto-advance between experiences.** Stop after each recap.
6. **Never run experiments on a non-quiet magnet4.** If `uptime` shows load > 4 before starting an A/B, wait or move to a different time window. Background load destroys A/B timing comparisons.
7. **Never use cached/inherited data splits across experiences without verifying the split seed.** If in doubt, regenerate.
8. **Do not propose changes that require external dependencies not already in the venv.** Stay within installed packages.

---

## 12. What to do if stuck

If you genuinely cannot proceed (a setup step keeps failing, an experience's hypotheses all abort, a tool isn't available, a path doesn't exist) — write the situation to the relevant `recap.md` (or `setup_status.md` for setup), output a clear `=== STUCK: <experience> — <one-line reason> ===` to terminal, and STOP.

Do not invent workarounds that might silently change what you're measuring. Stopping cleanly with a clear status note is always better than a misleading result.

---

## 13. Start

Begin with Section 4 (setup phase). Do not ask for confirmation. Do not summarize this file back. Just start.
