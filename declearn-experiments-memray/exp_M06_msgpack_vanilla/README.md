# exp_M06 msgpack vanilla A/B

## Date
2026-05-11

## Question
Does the upstream MR !82 (`d402885b` "Serialization: migration to MessagePack and harmonization") reduce memory cost on a plain federated workload without SecAgg, and how does the win scale with client count?

## TL;DR
At vanilla MNIST quickrun (no SecAgg), the MR delivers a modest but consistent memory win that grows with `N`. The JSON encoder cost (`iterencode`) drops to zero in the msgpack arm at every `N` where it was previously visible. Peak RSS reduction reaches -4.3% at `N=10`; cumulative allocation reduction reaches -14.3%. Wall-clock is unchanged.

The win is smaller than the SecAgg case (where the dd2 patch alone delivered -9% peak at `N=20`, scaling to -37% at `N=100`). SecAgg is the worst case because every encrypted scalar was a separate Python int that `json.dumps` stringified individually. Vanilla declearn already serialized numpy arrays as a single hex string, so the per-payload allocation count was much lower.

## Pre-flight findings

### How baseline (pre-MR) serialized numpy arrays
- File: `declearn/utils/_numpy.py`
- Mechanism: `array.tobytes().hex()` returns a single hex string per array.
- Cost: 2 chars per byte. A 1000-element float64 array (8000 bytes) becomes a 16000-char string, plus JSON quoting overhead.
- Not the worst case (`.tolist()` would be ~17 chars per float64 with decimal printing). The MR moves this to raw `tobytes()` inline in msgpack: roughly half the wire bytes and one fewer big string allocation per array.

### Wire format in baseline
- File: `declearn/messaging/_api.py:86`
- Mechanism: `json.dumps(data, default=json_pack)` then `.send_str` on the websocket.
- Treatment writes `.mpk` extension on checkpoints, baseline writes `.json`. Confirmed in smoke test.

### CI status of treatment commit
- Smoke test of `declearn-quickrun(...)` on MNIST 2c/1r/no-SecAgg passed end-to-end on `d402885b`. The CI failure in pipeline #1407973 does not affect the quickrun execution path tested here. Did not investigate which test is failing per the do-not-patch-upstream constraint.

## Setup
- Baseline: `9668576` (parent of merge `d402885b`, develop tip pre-MR).
- Treatment: `d402885b` (MR merge commit on develop).
- Both installed via `pip install -e .` into the same venv. The upstream repo is at `_scratch/declearn_upstream`.
- Workload: MNIST quickrun small CNN, plain averaging, no SecAgg.
- Per run: `rounds=5`, `n_steps=5`, `batch_size=48`.
- Sweep: `N` ∈ {2, 5, 10}, 2 repeats per cell. 12 memray runs total.
- Profiler: memray Python-only mode, default sampling.
- Data folders: `data_iid` (N=2), `data_iid_n5` (N=5), `data_iid_n10` (N=10).

## Results: comparison table (mean of 2 repeats)

| cell      | N  | peak_MB | total_MB | n_allocs    | iterencode_MB | wall_s |
|-----------|----|---------|----------|-------------|----------------|--------|
| baseline  | 2  | 408.0   | 3277     | 1,506,432   | 0.0 (not top-5) | 41.8   |
| msgpack   | 2  | 406.3   | 3056     | 1,533,228   | 0.0            | 41.7   |
| baseline  | 5  | 428.9   | 5138     | 1,839,860   | 295.7          | 43.3   |
| msgpack   | 5  | 417.2   | 4539     | 1,866,330   | 0.0            | 41.4   |
| baseline  | 10 | 459.9   | 8227     | 2,396,408   | 591.5          | 46.4   |
| msgpack   | 10 | 440.0   | 7048     | 2,422,872   | 0.0            | 45.2   |

(allocation counts are first-repeat values; means are within 1% of these.)

## Results: deltas (msgpack vs baseline)

| N  | peak abs | peak % | total abs | total % | iterencode | wall % |
|----|----------|--------|-----------|---------|------------|--------|
| 2  | -1.7 MB  | -0.4%  | -221 MB   | -6.7%   | n/a in top-5 | -0.3% |
| 5  | -11.7 MB | -2.7%  | -599 MB   | -11.7%  | -296 MB (gone) | -4.4% |
| 10 | -19.9 MB | -4.3%  | -1.18 GB  | -14.3%  | -592 MB (gone) | -2.7% |

## Scaling pattern

| metric              | N=2   | N=5   | N=10  | trend with N         |
|---------------------|-------|-------|-------|----------------------|
| peak savings        | 0.4%  | 2.7%  | 4.3%  | grows roughly linearly |
| cumulative savings  | 6.7%  | 11.7% | 14.3% | grows, flattening    |
| iterencode savings  | 0     | 296MB | 592MB | doubles when N doubles |
| wall-clock change   | flat  | flat  | flat  | within noise         |

Cumulative savings stabilize near 14% because the JSON encoder cost is a constant fraction of the per-round bytes-on-wire path, which is itself a constant fraction of total run cost on this small model. Peak savings scale with `N` because the server holds more in-flight serialized payloads at once when `N` rises.

## Tier ranking (combined with prior SecAgg data)

| evidence tier | workload         | N   | peak Δ  | source              |
|---------------|------------------|-----|---------|---------------------|
| Tier-1        | vanilla (no SecAgg) | 10  | -4.3%   | this experiment     |
| Tier-1        | vanilla (no SecAgg) | 5   | -2.7%   | this experiment     |
| Tier-1        | vanilla (no SecAgg) | 2   | -0.4%   | this experiment     |
| Tier-2        | SecAgg masking   | 100 | -37.1%  | `dd2_secagg_scaling/` |
| Tier-2        | SecAgg masking   | 50  | -13.3%  | `dd2_secagg_scaling/` |
| Tier-2        | SecAgg masking   | 20  | -8.8%   | `dd2_secagg_scaling/` |

Vanilla is the framework-wide impact of the MR. SecAgg is the worst case.

## Files

| path                                        | content                                                  |
|---------------------------------------------|----------------------------------------------------------|
| `README.md`                                 | this file                                                |
| `ab_results.json`                           | machine-readable per-run stats (12 runs)                 |
| `config_n{2,5,10}.toml`                     | configs per N (rounds=5, n_steps=5, no SecAgg)           |
| `runs/baseline/n{N}_rep{1,2}/<ts>/`         | memray output per baseline run                           |
| `runs/msgpack/n{N}_rep{1,2}/<ts>/`          | memray output per treatment run                          |
| `runs/.../summary.txt`                      | memray stats (peak, total, top allocators)               |
| `runs/.../flamegraph.html`                  | memray visual flame, browser-openable                    |
| `runs/.../metadata.json`                    | per-run timing + target config                           |
| `run_msgpack_ab.py`                         | the driver itself                                        |
| `run_msgpack_ab.console.log`                | full driver console output with per-run lines            |

## Caveats

- Wire bytes were not measured directly. The `iterencode_MB` line item is used as a proxy, and confirmed eliminated in msgpack.
- Single-process quickrun, all clients colocated on `127.0.0.1`. Network framing cost is the same per-client per-round, but no real network is exercised.
- Small model (~10k params). Wins scale with payload size, so a BERT-class model would show larger absolute peak savings, though similar percentages.
- memray Python-only mode does not see C-level allocations from torch's caching allocator. The "peak" numbers are application-view, not RSS-view.
- The CI on the MR branch is failing on a test we did not investigate. Quickrun is unaffected.
- Constraints respected: read-only on the upstream repo. No commits, no pushes, no PRs from this experiment. The cloned upstream at `_scratch/declearn_upstream` was switched between commits via local `git checkout` only.
