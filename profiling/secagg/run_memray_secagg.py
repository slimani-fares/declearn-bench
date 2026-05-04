"""Profile the BASELINE SecAgg runner under memray at one or more N values.

Sister tool to run_pyspy_secagg.py. Same workload, same config, same data
split, same OMP/MKL caps — only difference is the profiler. Outputs are
*interpretable* directly:
  - memray_flamegraph.html  (interactive D3 flamegraph, open in any browser)
  - memray_stats.txt        (text summary: total/peak heap, top allocators)
  - metadata.json
The raw memray .bin is deleted by default because it's large (~hundreds
of MB at N=10). Pass --keep-bin to retain it for regenerating extra
reports (memray summary, memray tree, memray table) later.

Memory protocol (per CLAUDE.md):
  - Python-only mode (no --native); native mode requires unstripped binaries.
  - --follow-fork ON so subprocesses (none expected here, but defensive).
  - Three independent dimensions to inspect: peak heap, total allocations,
    retained memory.
"""
import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime


CONFIG = "configs/secagg/config_fedavg_torch_secagg.toml"
RUNNER = "profiling/secagg/runner.py"
DATA_ROOT = "examples/mnist_quickrun"
DATA_DIR = os.path.join(DATA_ROOT, "data_iid")
SPLIT_SEED = 42
SPLIT_SCHEME = "iid"

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "2")


def ensure_data(n_clients: int):
    """Re-split MNIST for n_clients shards. Wipes data_iid/ first to avoid
    leftover client_* subdirs from a previous (larger) N."""
    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    subprocess.run([
        "declearn-split",
        "--folder", DATA_ROOT,
        "--n_shards", str(n_clients),
        "--scheme", SPLIT_SCHEME,
        "--seed", str(SPLIT_SEED),
    ], check=True)


def profile_one(n_clients: int, results_root: str, keep_bin: bool, variant: str):
    out_dir = os.path.join(results_root, f"N={n_clients}")
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, "memray.bin")
    flamegraph_path = os.path.join(out_dir, "memray_flamegraph.html")
    stats_path = os.path.join(out_dir, "memray_stats.txt")

    # memray refuses to overwrite an existing output file.
    if os.path.exists(bin_path):
        os.remove(bin_path)

    ensure_data(n_clients)

    print(f"\n=== Memray-profiling N={n_clients} ({variant}) ===")
    proc = subprocess.run([
        "memray", "run",
        "--output", bin_path,
        "--follow-fork",
        "python", RUNNER,
        "--config", CONFIG,
        "--n_clients", str(n_clients),
    ])

    if not os.path.exists(bin_path):
        raise RuntimeError(
            f"memray failed for N={n_clients} (exit {proc.returncode}); "
            f"no bin file at {bin_path}"
        )

    print("  Generating flamegraph...")
    subprocess.run([
        "memray", "flamegraph", "--output", flamegraph_path, bin_path,
    ], check=True)

    print("  Generating text stats...")
    with open(stats_path, "w") as f:
        subprocess.run(["memray", "stats", bin_path], stdout=f, check=True)

    if not keep_bin:
        os.remove(bin_path)
        print(f"  Removed {bin_path} (use --keep-bin to retain)")

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "n_clients": n_clients,
            "config": CONFIG,
            "split_seed": SPLIT_SEED,
            "split_scheme": SPLIT_SCHEME,
            "tool": "memray",
            "variant": variant,
        }, f, indent=2)

    print(f"Saved: {flamegraph_path}")
    print(f"Saved: {stats_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--keep-bin", action="store_true",
                        help="Keep the raw memray .bin (default: delete after reports)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    results_root = os.path.join("profiling/secagg/results-memray", timestamp)
    os.makedirs(results_root, exist_ok=True)

    for n in args.n_clients:
        profile_one(n, results_root, args.keep_bin, variant="baseline")

    print(f"\nAll done. Results in: {results_root}")


if __name__ == "__main__":
    main()
