"""py-spy profiling runner for the declearn-bench ASV classes.

Usage:
    python profiling/run_pyspy.py --class fairness --param config_fedavg_torch_fairbatch.toml --n-clients 5
    python profiling/run_pyspy.py --class scaffold --n-clients 5
    python profiling/run_pyspy.py --class dp --n-clients 3
    python profiling/run_pyspy.py --class secagg --n-clients 3 [--rate 200]

Runs `b.setup(...)` in-parent (unprofiled — materializes data) then spawns
`py-spy record -- python bench_target.py ...`. Writes a speedscope JSON +
metadata.json under results/pyspy/<tag>/<ts>/.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Make `benchmarks` importable when running this script directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import (  # noqa: E402
    BackendsBenchmark,
    DPBenchmark,
    FairnessBenchmark,
    RegularizersBenchmark,
    ScaffoldBenchmark,
    SecAggBenchmark,
)

CLASSES = {
    "backends": BackendsBenchmark,
    "regularizers": RegularizersBenchmark,
    "scaffold": ScaffoldBenchmark,
    "dp": DPBenchmark,
    "secagg": SecAggBenchmark,
    "fairness": FairnessBenchmark,
}

DEFAULT_PYTHON = os.path.expanduser("~/.venvs/declearn311/bin/python")
DEFAULT_PYSPY = os.path.expanduser("~/.venvs/declearn311/bin/py-spy")
REPO_ROOT = Path(__file__).resolve().parent.parent


def axis_choices(cls, name):
    """Return the list of valid values for a given param_names entry."""
    params = cls.params
    idx = cls.param_names.index(name)
    # Single-axis class: params is a flat list.
    if not isinstance(params[0], list):
        return params
    # Multi-axis class: params is a list-of-lists.
    return params[idx]


def validate(class_name, param, n_clients):
    if class_name not in CLASSES:
        raise SystemExit(f"Unknown class {class_name!r}. Choices: {sorted(CLASSES)}")
    cls = CLASSES[class_name]
    if not hasattr(cls, "param_names"):
        if param is not None or n_clients is not None:
            raise SystemExit(f"Class {class_name!r} takes no params.")
        return
    supplied = {"config": param, "n_clients": n_clients}
    for name in cls.param_names:
        value = supplied[name]
        if value is None:
            raise SystemExit(
                f"Class {class_name!r} requires --{name.replace('_', '-')}. "
                f"Choices: {axis_choices(cls, name)}"
            )
        valid = [str(v) for v in axis_choices(cls, name)]
        if str(value) not in valid:
            raise SystemExit(
                f"{name}={value!r} is not in {class_name}.{name} axis: {valid}"
            )
    # Reject unexpected flags.
    for name, value in supplied.items():
        if name not in cls.param_names and value is not None:
            raise SystemExit(f"Class {class_name!r} has no axis {name!r}.")


def run_setup(class_name, param, n_clients):
    cls = CLASSES[class_name]
    b = cls()
    if not hasattr(cls, "param_names"):
        b.setup()
        return
    mapping = {"config": param, "n_clients": n_clients}
    positional = [mapping[name] for name in cls.param_names]
    b.setup(*positional)


def capture_version(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip() or result.stderr.strip()


def build_tag(class_name, param, n_clients):
    parts = [class_name]
    if param is not None:
        parts.append(Path(param).stem)
    if n_clients is not None:
        parts.append(f"n{n_clients}")
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--class", dest="class_name", required=True,
                        choices=sorted(CLASSES))
    parser.add_argument("--param", default=None,
                        help="TOML filename for classes whose axis includes 'config'.")
    parser.add_argument("--n-clients", dest="n_clients", default=None,
                        help="Client count for classes whose axis includes 'n_clients'.")
    parser.add_argument("--rate", type=int, default=100,
                        help="py-spy sampling rate in Hz (default: 100).")
    parser.add_argument("--python", default=DEFAULT_PYTHON,
                        help=f"Python interpreter (default: {DEFAULT_PYTHON}).")
    parser.add_argument("--pyspy", default=DEFAULT_PYSPY,
                        help=f"py-spy binary (default: {DEFAULT_PYSPY}).")
    args = parser.parse_args()

    validate(args.class_name, args.param, args.n_clients)

    # Output layout.
    tag = build_tag(args.class_name, args.param, args.n_clients)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = REPO_ROOT / "profiling" / "results" / "pyspy" / tag / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    speedscope = out_dir / "pyspy_speedscope.json"
    metadata = out_dir / "metadata.json"

    # setup() in-parent (idempotent, unprofiled).
    run_setup(args.class_name, args.param, args.n_clients)

    # Build the py-spy invocation. Children are launched and attached
    # automatically by py-spy itself — no external PID wiring needed.
    target_cmd = [args.python, "profiling/bench_target.py",
                  "--class", args.class_name]
    if args.param is not None:
        target_cmd += ["--param", args.param]
    if args.n_clients is not None:
        target_cmd += ["--n-clients", str(args.n_clients)]
    cmd = [
        args.pyspy, "record",
        "--format", "speedscope",
        "--rate", str(args.rate),
        "-o", str(speedscope),
        "--",
        *target_cmd,
    ]

    print(f"Running: {' '.join(cmd)}\n")
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    duration = time.perf_counter() - start

    # Versions.
    declearn_version = capture_version(
        [args.python, "-c", "import declearn; print(declearn.__version__)"]
    )
    pyspy_version = capture_version([args.pyspy, "--version"])
    python_version = capture_version([args.python, "--version"])

    metadata_doc = {
        "timestamp": ts,
        "class": args.class_name,
        "param": args.param,
        "n_clients": args.n_clients,
        "sampling_rate_hz": args.rate,
        "wall_clock_seconds": round(duration, 2),
        "return_code": result.returncode,
        "status": "success" if result.returncode == 0 else "failed",
        "python": args.python,
        "python_version": python_version,
        "declearn_version": declearn_version,
        "py_spy_version": pyspy_version,
    }
    with open(metadata, "w") as f:
        json.dump(metadata_doc, f, indent=2)

    print(f"\n=== py-spy profile complete ===")
    suffix = []
    if args.param:
        suffix.append(args.param)
    if args.n_clients:
        suffix.append(f"n={args.n_clients}")
    print(f"Class:    {args.class_name}" + (f" ({', '.join(suffix)})" if suffix else ""))
    print(f"Duration: {metadata_doc['wall_clock_seconds']}s")
    print(f"Status:   {metadata_doc['status']}")
    print(f"Output:   {out_dir}")


if __name__ == "__main__":
    main()
