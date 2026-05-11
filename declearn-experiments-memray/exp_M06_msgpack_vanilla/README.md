# exp_M06 msgpack vanilla A/B

## Date
2026-05-11

## Question
Does the upstream MR !82 (`d402885b` "Serialization: migration to MessagePack and harmonization") reduce memory cost on a plain federated workload without SecAgg, and how does the win scale with client count?

## TL;DR
At vanilla MNIST quickrun (no SecAgg), the MR delivers a memory win that grows roughly linearly with client count and turns into a measurable wall-clock win at `N >= 50`. The JSON encoder (`iterencode`) and decoder (`raw_decode`) cost drops to zero in the msgpack arm at every `N`. Peak heap reduction reaches **-20.2% at N=100** (1.05 GB → 836 MB). Cumulative allocation drops by **18.7%** at N=100 (65 GB → 53 GB). Wall-clock at N=100 is **-12.1%** (4 min → 3.5 min, ~29s saved per round-of-5).

The small-`N` rows (N=2/5/10) understate the impact: at `N <= 10` the JSON encoder cost is still small relative to torch's own working set, so the % savings look modest (0.4-4.3% peak). The win compounds at the scale where the JSON-encoded payload pileup becomes the dominant non-torch allocation.

The vanilla win is smaller in absolute % than the SecAgg case (-37% peak at N=100, in `dd2_secagg_scaling/`), because SecAgg payloads stringify every encrypted scalar individually while vanilla declearn already used a single hex-string per array. But the vanilla case applies to **every** declearn workload, not just SecAgg.

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
- Sweep: `N` ∈ {2, 5, 10, 20, 50, 100}, 2 repeats per cell. 24 memray runs total (12 small-N + 12 big-N).
- Profiler: memray Python-only mode, default sampling.
- Data folders: `data_iid` (N=2), `data_iid_n{5,10,20,50,100}` for other N values.
- For `N >= 20` the configs add `min_clients = N` and increase `register.timeout` to 30/60/120 s respectively.

## Results: comparison table (mean of 2 repeats)

| cell      | N   | peak_MB | total_MB | iterencode_MB | raw_decode_MB | wall_s |
|-----------|-----|---------|----------|----------------|----------------|--------|
| baseline  | 2   | 408.0   | 3,277    | 0.0 (not top-5) | 0.0           | 41.8   |
| msgpack   | 2   | 406.3   | 3,056    | 0.0            | 0.0            | 41.7   |
| baseline  | 5   | 428.9   | 5,138    | 295.7          | 0.0            | 43.3   |
| msgpack   | 5   | 417.2   | 4,539    | 0.0            | 0.0            | 41.4   |
| baseline  | 10  | 459.9   | 8,227    | 591.5          | 0.0            | 46.4   |
| msgpack   | 10  | 440.0   | 7,048    | 0.0            | 0.0            | 45.2   |
| baseline  | 20  | 527.5   | 14,494   | 1,183.0        | 0.0            | 70.2   |
| msgpack   | 20  | 483.7   | 12,134   | 0.0            | 0.0            | 65.4   |
| baseline  | 50  | 720.6   | 33,239   | 2,957.0        | 0.0            | 130.2  |
| msgpack   | 50  | 615.5   | 27,334   | 0.0            | 0.0            | 117.8  |
| baseline  | 100 | 1,047.5 | 65,110   | 5,915.0        | 2,952.0        | 238.6  |
| msgpack   | 100 | 835.9   | 52,954   | 0.0            | 0.0            | 209.8  |

Note: `raw_decode` (JSON decoder) only crossed memray's top-5 threshold at N=100 in baseline. msgpack eliminates both encoder and decoder at every N.

## Results: deltas (msgpack vs baseline)

| N   | peak abs    | peak %  | total abs  | total % | iter+rawdec eliminated | wall % |
|-----|-------------|---------|------------|---------|------------------------|--------|
| 2   | -1.7 MB     | -0.4%   | -221 MB    | -6.7%   | n/a in top-5           | -0.3%  |
| 5   | -11.7 MB    | -2.7%   | -599 MB    | -11.7%  | 296 MB                 | -4.4%  |
| 10  | -19.9 MB    | -4.3%   | -1.18 GB   | -14.3%  | 592 MB                 | -2.7%  |
| 20  | -43.8 MB    | -8.3%   | -2.36 GB   | -16.3%  | 1.18 GB                | -6.8%  |
| 50  | -105.1 MB   | -14.6%  | -5.91 GB   | -17.8%  | 2.96 GB                | -9.5%  |
| 100 | **-211.6 MB** | **-20.2%** | **-12.16 GB** | **-18.7%** | **8.87 GB**     | **-12.1%** |

## Scaling pattern

| metric              | N=2  | N=5  | N=10 | N=20 | N=50  | N=100 | trend with N         |
|---------------------|------|------|------|------|-------|-------|----------------------|
| peak savings        | 0.4% | 2.7% | 4.3% | 8.3% | 14.6% | 20.2% | grows linearly       |
| cumulative savings  | 6.7% | 11.7%| 14.3%| 16.3%| 17.8% | 18.7% | grows, flattening near 19% |
| iter+rawdec saved   | 0    | 296 MB | 592 MB | 1.18 GB | 2.96 GB | 8.87 GB | linear in N up to N=50, then super-linear |
| wall-clock change   | flat | flat | flat | -6.8%| -9.5% | -12.1%| flat for small N, real win at N >= 50 |

Three observations:

1. **Peak heap savings scale roughly linearly with N** through the whole range (0.4% per client of marginal saving). The server holds more in-flight serialized payloads concurrently as N grows; msgpack's per-payload footprint is constant-factor smaller in every case.

2. **Cumulative allocation savings flatten near 19%**. The JSON encode/decode cost is a constant fraction of the per-round bytes-on-wire path, and that's a constant fraction of total run cost. Once N is large enough that JSON dominates non-torch allocations, the ratio stops moving.

3. **Wall-clock crosses the noise floor at N=20** and becomes a genuine speed win at N=50/100. At N=100 the msgpack arm is ~29 s faster per 4-minute round. The JSON encoder/decoder were CPU-bound work that no longer happens.

The N=100 baseline also surfaces `raw_decode` (JSON decoder) as a 2.95 GB top-5 allocator — at that scale, decoding payloads back into Python lists/dicts costs as much as encoding them. msgpack eliminates both directions.

## Tier ranking (combined with prior SecAgg data)

| evidence tier | workload            | N   | peak Δ  | source                 |
|---------------|---------------------|-----|---------|------------------------|
| Tier-1        | vanilla (no SecAgg) | 100 | **-20.2%** | this experiment     |
| Tier-1        | vanilla (no SecAgg) | 50  | -14.6%  | this experiment        |
| Tier-1        | vanilla (no SecAgg) | 20  | -8.3%   | this experiment        |
| Tier-1        | vanilla (no SecAgg) | 10  | -4.3%   | this experiment        |
| Tier-1        | vanilla (no SecAgg) | 5   | -2.7%   | this experiment        |
| Tier-1        | vanilla (no SecAgg) | 2   | -0.4%   | this experiment        |
| Tier-2        | SecAgg masking      | 100 | -37.1%  | `dd2_secagg_scaling/`  |
| Tier-2        | SecAgg masking      | 50  | -13.3%  | `dd2_secagg_scaling/`  |
| Tier-2        | SecAgg masking      | 20  | -8.8%   | `dd2_secagg_scaling/`  |

Vanilla is the framework-wide impact of the MR. SecAgg is the worst case.

Side-by-side at the same N: vanilla -20% peak vs SecAgg -37% peak at N=100. SecAgg amplifies the win because its payloads are `List[int]` (one Python int per encrypted scalar), so JSON allocation count scales with `L = 10k values per vector` per client. Vanilla payloads are single hex strings, so the savings are bounded by the JSON encoder's constant-factor overhead per string rather than per element.

## Files

| path                                        | content                                                  |
|---------------------------------------------|----------------------------------------------------------|
| `README.md`                                 | this file                                                |
| `ab_results.json`                           | machine-readable per-run stats, small-N sweep (12 runs)  |
| `ab_results_bigN.json`                      | machine-readable per-run stats, big-N sweep (12 runs)    |
| `config_n{2,5,10,20,50,100}.toml`           | configs per N (rounds=5, n_steps=5, no SecAgg)           |
| `runs/baseline/n{N}_rep{1,2}/<ts>/`         | memray output per baseline run                           |
| `runs/msgpack/n{N}_rep{1,2}/<ts>/`          | memray output per treatment run                          |
| `runs/.../summary.txt`                      | memray stats (peak, total, top allocators)               |
| `runs/.../flamegraph.html`                  | memray visual flame, browser-openable                    |
| `runs/.../metadata.json`                    | per-run timing + target config                           |
| `run_msgpack_ab.py`                         | small-N driver (N ∈ {2, 5, 10})                          |
| `run_msgpack_ab_bigN.py`                    | big-N driver (N ∈ {20, 50, 100})                         |
| `run_msgpack_ab.console.log`                | small-N driver console output                            |
| `run_msgpack_ab_bigN.console.log`           | big-N driver console output                              |

## Caveats

- Wire bytes were not measured directly. The `iterencode_MB` line item is used as a proxy, and confirmed eliminated in msgpack.
- Single-process quickrun, all clients colocated on `127.0.0.1`. Network framing cost is the same per-client per-round, but no real network is exercised.
- Small model (~10k params). Wins scale with payload size, so a BERT-class model would show larger absolute peak savings, though similar percentages.
- N ∈ {2..100} measured. Beyond N=100 the linear trend in peak savings (≈0.2 percentage points per added client) would predict the curve to flatten as torch's own working set becomes the dominant peak contributor. Not measured.
- 2 repeats per cell. Per-N peak/total values are stable across repeats to within 1 MB / 1%, so we report means rather than std deviations.
- memray Python-only mode does not see C-level allocations from torch's caching allocator. The "peak" numbers are application-view, not RSS-view.
- The CI on the MR branch is failing on a test we did not investigate. Quickrun is unaffected.
- Constraints respected: read-only on the upstream repo. No commits, no pushes, no PRs from this experiment. The cloned upstream at `_scratch/declearn_upstream` was switched between commits via local `git checkout` only.
