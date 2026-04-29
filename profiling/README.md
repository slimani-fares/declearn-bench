# profiling

Profile a single declearn-bench ASV class with py-spy. The runner calls the
benchmark's `setup()` in-parent (so data materialization doesn't pollute the
flamegraph) then spawns py-spy, which launches and attaches to the actual
`time_quickrun()` subprocess itself.

## Usage

Every benchmark class has an `n_clients` axis. Classes that also vary a TOML
config expose a `config` axis too.

```bash
# Single-axis classes (n_clients only)
python profiling/run_pyspy.py --class scaffold --n-clients 5
python profiling/run_pyspy.py --class dp       --n-clients 3
python profiling/run_pyspy.py --class secagg   --n-clients 3

# Two-axis classes (config × n_clients)
python profiling/run_pyspy.py --class backends     --param config_fedavg_torch.toml         --n-clients 5
python profiling/run_pyspy.py --class regularizers --param config_fedavg_torch_lasso.toml   --n-clients 3
python profiling/run_pyspy.py --class fairness     --param config_fedavg_torch_fairbatch.toml --n-clients 5
```

Results land in `profiling/results/pyspy/<tag>/<timestamp>/` as
`pyspy_speedscope.json` + `metadata.json`. Drop the speedscope JSON into
<https://www.speedscope.app> to view. `<tag>` is built from `<class>[_<config_stem>]_n<N>`.

## Flags

- `--rate 200` — bump sampling rate (default 100 Hz).
- `--python`, `--pyspy` — override interpreter / py-spy binary
  (default: `~/.venvs/declearn311/`).

## Supported classes

| Class | Axes |
|---|---|
| `backends` | `config` (torch, tensorflow) × `n_clients` (3, 5) |
| `regularizers` | `config` (lasso, ridge, fedprox) × `n_clients` (3, 5) |
| `fairness` | `config` (fairbatch, fairfed, fairgrad) × `n_clients` (3, 5) |
| `scaffold` | `n_clients` (3, 5) |
| `dp` | `n_clients` (3, 5) |
| `secagg` | `n_clients` (3, 5) — masking variant only |
