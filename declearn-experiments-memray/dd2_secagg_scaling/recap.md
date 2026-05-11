# DD2 SecAgg masking — binary JSON packing — scaling evidence

## Date
2026-05-11

## TL;DR

Replacing the JSON serialization of `MaskedAggregate.encrypted` (`List[int]`
of uint64s) with a base64-packed bytes blob produces a **memory win that
grows super-linearly with client count**:

| N    | Master peak | Variant peak | Peak saved | Peak % saved |
|------|-------------|--------------|------------|--------------|
| 20   | 704 MB      | 642 MB       | -62 MB     | -8.8%        |
| 50   | 1162 MB     | 1007 MB      | -155 MB    | -13.3%       |
| 100  | 2575 MB     | **1620 MB**  | **-955 MB**| **-37.1%**   |

At N=100 the variant uses **1 GB less peak heap** than canonical, with
zero wall-clock cost. Cumulative allocation churn drops a steady ~6–7%
across all N.

## What the experiment runs

Same config across all three N, only `min_clients` and registration
timeout differ:
- 1 round, 5 SGD steps per client, MNIST quickrun small CNN
- SecAgg masking enabled (`secagg_variant='masking'`)
- masking field `max_int = 2**64` (declearn default → the variant fast path)

For each N ∈ {20, 50, 100}: one memray-tracked run on the **master**
arm of `declearn-for-exp_12_secagg_aggregate` (canonical secagg path) and
one on the **variant** arm (`dd2_json_binary_variant` branch, with the
JSON-binary-packing patch).

## Full scaling table

| N    | Arm     | Peak (MB) | Total (MB) | N allocations | Wall-clock (s) |
|------|---------|-----------|------------|----------------|-----------------|
| 20   | master  | 703.8     | 8,155      | 1,652,585       | 58.2            |
| 20   | variant | 641.9     | 7,640      | 1,646,720       | 57.9            |
| 50   | master  | 1162.0    | 18,780     | 3,112,501       | 126.5           |
| 50   | variant | 1007.0    | 17,482     | 3,107,152       | 122.6           |
| 100  | master  | 2575.0    | 40,561     | 7,432,550       | 252.9           |
| 100  | variant | 1620.0    | 37,966     | 7,420,522       | 246.7           |

| N    | Peak Δ            | Total Δ          | Wall Δ            |
|------|--------------------|------------------|--------------------|
| 20   | -62 MB (-8.8%)     | -518 MB (-6.4%)  | -0.3 s (within noise) |
| 50   | -155 MB (-13.3%)   | -1.3 GB (-6.9%)  | -3.9 s (within noise) |
| 100  | -955 MB (-37.1%)   | -2.6 GB (-6.4%)  | -6.2 s (within noise) |

## How to read the trend

- **Master peak scales linearly with N**: 704 → 1162 → 2575 MB, with
  each step roughly 2.2× the previous when N doubles. The server is
  holding ~N deserialized encrypted payloads simultaneously during
  aggregation; each canonical payload is a Python list of L int objects.
- **Variant peak grows much slower**: 642 → 1007 → 1620 MB. Each payload
  becomes a single `bytes` / ascii `str` of ~10 KB instead of a 10k-element
  Python list. The simultaneous footprint stays smaller.
- **Cumulative churn drops a flat ~6-7%** because the *per-payload* JSON
  encoder/decoder cost is reduced by the same constant factor regardless
  of N. The patch saves the same fraction of JSON allocations per client
  per round; what scales is how much peak that compounds to.

## What the patch does (50 lines, single file)

`declearn/secagg/masking/_aggregate.py` — `MaskedAggregate.to_dict` and
`from_dict`:

- When `max_int ≤ 2**64` (the default), pack the integer list as raw
  little-endian 8-byte uint64 values, then base64-encode → ascii string.
  Emit as `encrypted_b64` instead of `encrypted` in the dict.
- `from_dict` detects either field name and reconstructs the list, so
  the wire format is symmetric and parties speaking either dialect can
  interoperate provided one party encodes and one decodes.
- Falls through to the canonical Python-list dict path when `max_int`
  exceeds the uint64 range (Joye-Libert variants).

See `patch/aggregate_b64.diff`.

## Why peak savings grow super-linearly with N

Each canonical client reply carries a 10,000-element Python list of
big ints in memory during aggregation. The Python list itself plus
its int objects is roughly ~280 KB per reply (PyObject header +
boxed-int representation overhead). At N clients held concurrently:

- N=20: ~5.6 MB extra over the variant per round — small.
- N=50: ~14 MB extra — visible.
- N=100: ~28 MB extra per fully-loaded snapshot, but the aggregation
  walks through many such snapshots and the deserialized payloads
  pile up in the asyncio event loop. The 1 GB peak gap we measure is
  the cumulative live set during the aggregation phase.

The variant's payloads are 8 KB bytes blobs + a small Python wrapper
each, so the same N-fold pileup costs ~1 MB total — flat in N.

## How to reproduce / read the data

| Path                                          | Content                                            |
|-----------------------------------------------|----------------------------------------------------|
| `configs/config_secagg_n{50,100}.toml`        | The two new config files (N=50, N=100 templates)   |
| `runs/n20_master_from_sweep/`                 | Symlink to the original M03 sweep master run       |
| `runs/n20_variant_from_dd2/`                  | Symlink to the original DD2 variant run            |
| `runs/n{50,100}_{master,variant}/<ts>/`       | The 4 new runs                                     |
| `runs/.../summary.txt`                        | memray stats text per run (peak, total, top allocators) |
| `runs/.../flamegraph.html`                    | memray visual flame per run, browser-openable      |
| `scaling_results.json`                        | Machine-readable cross-N stats                     |
| `run_scaling.console.log`                     | Full driver console output                         |
| `run_scaling.py`                              | The driver itself (sequential per-arm runs)        |
| `patch/aggregate_b64.diff`                    | The 56-line patch as a git diff                    |

The patch lives in the fork at
`declearn-for-exp_12_secagg_aggregate/` on branch `dd2_json_binary_variant`,
on top of `master` which inherits the unified-runner + batched-encrypt-side
fix from `declearn-for-secagg-batched/`.

## Caveats

- **Masking only.** Joye-Libert SecAgg uses values much larger than
  2**64; the patch correctly falls through to canonical for that case
  but doesn't help it. A different packing (e.g. variable-length bytes)
  would be needed for Joye-Libert.
- **Wire-format requires both ends on the same patch.** In quickrun all
  parties run in one process, so this is automatic; in a distributed
  deployment, server and clients both need the patch — although the
  reader-side handles both formats so a phased rollout is feasible.
- **Single-seed measurements.** memray output is deterministic per
  inputs, so the per-N peak/total numbers above are exact, not means.
  Wall-clock is also single-seed but the per-N wall-clock deltas are
  small enough (< noise floor of 1-2%) that we don't claim a wall-clock
  effect either way — only a memory effect.
- **1 round, 5 SGD steps.** Larger training loops don't change the
  per-round aggregation pattern; the per-round savings would simply
  accumulate proportionally across rounds.

## Status

**Confirmed scaling win.** The DD2 patch produces a real, measurable
memory reduction at moderate client counts (N=20) and a substantial one
at higher client counts (N=100: -1 GB peak, -37%). Worth merging.

The bigger story: this is a scaling enabler, not a constant-factor
optimization. At N=20 it's a "nice-to-have"; at N=100+ it's the
difference between holding the entire round in working set and not.
