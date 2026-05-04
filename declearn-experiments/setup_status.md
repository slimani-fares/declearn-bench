# Setup status

STATUS: READY

Date: 2026-04-30T21:10+02:00
Host: magnet4.lille.inria.fr
Operator: Claude (autonomous loop), Fares interactive session

## Environment

- venv: `~/.venvs/declearn313` (Python 3.13.13)
- declearn: 2.8.0 (editable install at `~/declearn-bench/declearn/`)
- torch: 2.11.0+cu130 (CPU-only)
- py-spy: 0.4.2
- Cluster load at start: `0.19 0.11 0.23` (well under the 4.0 floor in Section 11.6)
- 16 cores, 188 GiB RAM available

## Forks created

None yet. Per Section 4.3, forks are created on demand per-experience when an
optimization is proposed. No fork is required for the baseline observation
profiles in Section 6.1.

## Configs created

For each of the 9 experiences, the following exist under
`~/declearn-bench/declearn-experiments/<exp_id>/`:

- `config_baseline.toml` — vanilla FedAvg-torch (copied from
  `configs/backends/config_fedavg_torch.toml`). Identical across all 9
  experiences as expected by Section 4.2.
- `config_variant.toml` — feature-on (copied from the matching
  `configs/<feature>/` TOML).

Specific paths:

| exp | variant source |
|---|---|
| 01 | `configs/regularizers/config_fedavg_torch_lasso.toml` |
| 02 | `configs/regularizers/config_fedavg_torch_ridge.toml` |
| 03 | `configs/regularizers/config_fedavg_torch_fedprox.toml` |
| 04 | `configs/dp/config_fedavg_torch_dp.toml` |
| 05 | `configs/scaffold/config_fedavg_torch_scaffold.toml` |
| 06 | `configs/secagg/config_fedavg_torch_secagg.toml` (masking variant — see caveats) |
| 07 | `configs/fairness/config_fedavg_torch_fairgrad.toml` |
| 08 | `configs/fairness/config_fedavg_torch_fairbatch.toml` |
| 09 | `configs/fairness/config_fedavg_torch_fairfed.toml` |

The smoke-test config lives at `declearn-experiments/_setup/config_smoke.toml`
(2 clients, 1 round, vanilla FedAvg-torch — produced for Section 4.4).

## Smoke test result

PASS.

- Tool used: `profiling/run_profile.py` (newly created — see "Notes" below)
- Config: `declearn-experiments/_setup/config_smoke.toml`
- Data split: `declearn-split --folder examples/mnist_quickrun --n_shards 2 --scheme iid --seed 42`
- Wall-clock: 48.03 s (well under the 5-minute floor)
- py-spy samples: 1243 (rate 100 Hz)
- Speedscope JSON: `declearn-experiments/_setup/runs/smoke_canonical/2026-04-30_21-09-53/pyspy_speedscope.json` (337 kB, 1347 frames in `shared.frames`, 1243 weights/samples in `profiles[0]`)
- Final accuracy at round 1 (sanity): 0.83 across both clients — looks like a normal MNIST quickrun trajectory

## Notes / caveats

1. **`run_profile.py` was created from scratch** under `~/declearn-bench/profiling/run_profile.py`. The CLAUDE.md spec (Section 4.4) acknowledged it might need to be recreated and authorized doing so. It is a thin wrapper that takes any quickrun TOML and runs `declearn.quickrun._run.quickrun(config_path)` under py-spy, writing speedscope + metadata. Lives alongside the existing `run_pyspy.py` (which uses the ASV bench harness — see point 2 below).

2. **`benchmarks/__init__.py` is inconsistent with `profiling/run_pyspy.py`.** The committed `benchmarks/__init__.py` (251 lines) defines only `BackendsBenchmark`, `RegularizersBenchmark`, `ScaffoldBenchmark`, and `DPBenchmark`. But `run_pyspy.py` imports `SecAggBenchmark` and `FairnessBenchmark` too — those will fail at import time. A backup (`benchmarks/__init__.py.backup`, 488 lines) contains the missing classes plus a `_ensure_data_fair()` helper and a `_secagg_async_main()` runner. Diff is purely additive (238 additions, 0 changes to existing classes). I did NOT restore the backup because it's a meaningful state change and the file was deliberately renamed `.backup` — Fares should decide whether to restore. **Workaround: `run_profile.py` does not depend on the bench harness, so all 9 experiences can proceed via direct quickrun + py-spy without needing the SecAgg/Fairness ASV classes.**

3. **`run_pyspy.py` defaults to `~/.venvs/declearn311/`** but the working venv is `declearn313`. Pass `--python ~/.venvs/declearn313/bin/python --pyspy ~/.venvs/declearn313/bin/py-spy` if it ever gets fixed.

4. **`declearn` module exposes no `__version__` attribute.** `import declearn; declearn.__version__` raises `AttributeError`. Use `importlib.metadata.version('declearn')` instead. The `run_profile.py` metadata capture does this correctly. `run_pyspy.py` still uses the broken pattern.

5. **Joye-Libert SecAgg vs masking (exp_06).** The CLAUDE.md spec calls exp_06 `secagg_jl` (Joye-Libert). The existing `configs/secagg/config_fedavg_torch_secagg.toml` and the backup's `_secagg_async_main` are masking-only. The backup explicitly comments: *"Joye-Libert: parked — orders-of-magnitude slower on dense models."* The variant TOML I copied for exp_06 is the masking config; when exp_06 begins it will need to either (a) be adapted to JL by swapping `MaskingSecaggConfig*` → `JoyeLibertSecaggConfig*` per Section 5, or (b) be reframed as masking with a recap note explaining the JL/masking decision. **Defer this decision to exp_06 startup.**

6. **The unified runner patch (Section 5) has not been applied.** It is required for exp_06–09. The investigation reports under `~/declearn-bench/investigation_secagg_*.md` describe the patch but it lives in fork copies that haven't been created yet (per Section 4.3, forks are made on demand). When exp_06 begins, the first step is to fork `declearn/` into `declearn-for-exp_06_secagg_jl/` and apply the patch there.

7. **Data split.** A 2-shard iid split was created at `examples/mnist_quickrun/data_iid/` with seed 42 for the smoke test. Experiences 1–5 will likely use 3 or 5 shards (per Section 6.1: small scale = 2 clients, but the bench harness convention is 3/5). The `_ensure_data*` helpers in `benchmarks/__init__.py` create separate `data_iid_n<N>/` folders — those run on demand when the harness is invoked. For experiences using `run_profile.py` directly, the experience's recap should note which split was used and how it was generated.

## Next step (per Section 4.5)

CLAUDE.md authorizes proceeding directly into experience 1 without further
confirmation. The setup is healthy and the pipeline is verified end-to-end.

Suggested entry point for exp_01:

```bash
source ~/.venvs/declearn313/bin/activate
cd ~/declearn-bench
# observation profile per Section 6.1: 2 clients, 2 rounds, lasso variant
python profiling/run_profile.py --config declearn-experiments/exp_01_reg_lasso/config_variant.toml --tag exp_01_observation
```

The current variant TOML for exp_01 has `rounds = 2` already, which matches
the Section 6.1 default. Data split for n=2 already exists at
`examples/mnist_quickrun/data_iid/`. If we want a separate data folder per
experience to avoid stale state, regenerate per Section 11.7.
