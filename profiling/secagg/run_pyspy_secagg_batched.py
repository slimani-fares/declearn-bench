"""Profile the BATCHED SecAgg runner under py-spy at one or more N values.

Sister script to run_pyspy_secagg.py. Differences:
  * RUNNER points to profiling/secagg/runner_batched.py.
  * Output dir is profiling/secagg/results-batched/<timestamp>/ instead of
    .../results/<timestamp>/, so baseline and batched results don't collide.
  * metadata.json carries variant="batched".
Everything else (CONFIG, SPLIT_SEED, SPLIT_SCHEME, ensure_data, default
n_clients list, thread caps) is identical to the baseline driver — keep it
that way so the comparison stays valid.
"""
import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime


CONFIG = "configs/secagg/config_fedavg_torch_secagg.toml"
RUNNER = "profiling/secagg/runner_batched.py"
DATA_ROOT = "examples/mnist_quickrun"
DATA_DIR = os.path.join(DATA_ROOT, "data_iid")
SPLIT_SEED = 42
SPLIT_SCHEME = "iid"

# With N client coroutines each spawning BLAS threads, the default
# (one-thread-per-core) leads to crippling oversubscription on high-core nodes.
# Cap to 2 threads per coroutine; child subprocess inherits these.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "2")


def ensure_data(n_clients: int):
    """Re-split MNIST for n_clients shards.

    Wipes the existing data_iid/ directory first so leftover client_*
    subdirs from a previous (larger) N can't be picked up by
    parse_data_folder, which would silently spawn extra client coroutines.
    """
    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    subprocess.run([
        "declearn-split",
        "--folder", DATA_ROOT,
        "--n_shards", str(n_clients),
        "--scheme", SPLIT_SCHEME,
        "--seed", str(SPLIT_SEED),
    ], check=True)


def profile_one(n_clients: int, results_root: str):
    out_dir = os.path.join(results_root, f"N={n_clients}")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "pyspy_speedscope.json")

    ensure_data(n_clients)

    print(f"\n=== Profiling N={n_clients} (batched) ===")
    # py-spy can exit non-zero with ECHILD ("No child process") on Python 3.13
    # after successfully writing the speedscope file — asyncio's child-reaping
    # races with py-spy's wait(). Don't pass check=True; instead, treat the
    # presence of the speedscope file as the source of truth. If it's there,
    # the profile is good and we still want metadata.json next to it.
    proc = subprocess.run([
        "py-spy", "record",
        "--subprocesses",
        "--format", "speedscope",
        "-o", out_json,
        "--", "python", RUNNER,
        "--config", CONFIG,
        "--n_clients", str(n_clients),
    ])

    if not os.path.exists(out_json):
        raise RuntimeError(
            f"py-spy failed for N={n_clients} (exit {proc.returncode}); "
            f"no speedscope file at {out_json}"
        )

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "n_clients": n_clients,
            "config": CONFIG,
            "split_seed": SPLIT_SEED,
            "split_scheme": SPLIT_SCHEME,
            "variant": "batched",
        }, f, indent=2)

    if proc.returncode != 0:
        print(f"  (py-spy exited {proc.returncode}; speedscope written, continuing)")
    print(f"Saved: {out_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, nargs="+",
                        default=[2, 5, 10, 100])
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    results_root = os.path.join("profiling/secagg/results-batched", timestamp)
    os.makedirs(results_root, exist_ok=True)

    for n in args.n_clients:
        profile_one(n, results_root)

    print(f"\nAll done. Results in: {results_root}")


if __name__ == "__main__":
    main()
