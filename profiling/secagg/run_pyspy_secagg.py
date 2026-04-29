"""Profile the SecAgg runner under py-spy at one or more N values."""
import argparse
import json
import os
import subprocess
from datetime import datetime


CONFIG = "configs/secagg/config_fedavg_torch_secagg.toml"
RUNNER = "profiling/secagg/runner.py"
SPLIT_SEED = 42
SPLIT_SCHEME = "iid"


def ensure_data(n_clients: int):
    """Re-split MNIST for n_clients shards."""
    subprocess.run([
        "declearn-split",
        "--folder", "examples/mnist_quickrun",
        "--n_shards", str(n_clients),
        "--scheme", SPLIT_SCHEME,
        "--seed", str(SPLIT_SEED),
    ], check=True)


def profile_one(n_clients: int, results_root: str):
    out_dir = os.path.join(results_root, f"N={n_clients}")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "pyspy_speedscope.json")

    ensure_data(n_clients)

    print(f"\n=== Profiling N={n_clients} ===")
    subprocess.run([
        "py-spy", "record",
        "--subprocesses",
        "--format", "speedscope",
        "-o", out_json,
        "--", "python", RUNNER,
        "--config", CONFIG,
        "--n_clients", str(n_clients),
    ], check=True)

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "n_clients": n_clients,
            "config": CONFIG,
            "split_seed": SPLIT_SEED,
            "split_scheme": SPLIT_SCHEME,
        }, f, indent=2)

    print(f"Saved: {out_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, nargs="+",
                        default=[2, 5, 10, 100])
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    results_root = os.path.join("profiling/secagg/results", timestamp)
    os.makedirs(results_root, exist_ok=True)

    for n in args.n_clients:
        profile_one(n, results_root)

    print(f"\nAll done. Results in: {results_root}")


if __name__ == "__main__":
    main()
