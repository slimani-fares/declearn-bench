"""Generic config-driven profiling runner for declearn-bench experiments.

Usage:
    python profiling/run_profile.py --config <path-to-toml> [--tools pyspy] [--rate 100] [--out <dir>]

Wraps `declearn.quickrun._run.quickrun(config_path)` with py-spy. Writes a
speedscope JSON + metadata.json to <out>/<config_stem>/<timestamp>/.

This bypasses the benchmarks/ ASV harness (which currently has missing
SecAgg/Fairness benchmark classes — see setup_status.md). It accepts any
declearn quickrun TOML directly, so it works for all 9 experiences.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYTHON = sys.executable
DEFAULT_PYSPY = str(Path(sys.executable).parent / "py-spy")


def capture_version(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip() or result.stderr.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True,
                        help="Path to a declearn quickrun TOML config.")
    parser.add_argument("--tools", default="pyspy",
                        help="Comma-separated list of profilers (pyspy only for now).")
    parser.add_argument("--rate", type=int, default=100,
                        help="py-spy sampling rate in Hz (default: 100).")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: <config_dir>/runs).")
    parser.add_argument("--python", default=DEFAULT_PYTHON,
                        help=f"Python interpreter (default: {DEFAULT_PYTHON}).")
    parser.add_argument("--pyspy", default=DEFAULT_PYSPY,
                        help=f"py-spy binary (default: {DEFAULT_PYSPY}).")
    parser.add_argument("--tag", default=None,
                        help="Override tag in output path (default: config stem).")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    if tools != ["pyspy"]:
        raise SystemExit(f"Only --tools pyspy is supported (got {tools}).")

    tag = args.tag or config_path.stem
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(args.out) if args.out else (config_path.parent / "runs")
    out_dir = out_root / tag / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    speedscope = out_dir / "pyspy_speedscope.json"
    metadata_path = out_dir / "metadata.json"
    log_path = out_dir / "run.log"

    target = (
        "import asyncio; "
        "from declearn.quickrun._run import quickrun; "
        f"asyncio.run(quickrun({str(config_path)!r}))"
    )
    cmd = [
        args.pyspy, "record",
        "--format", "speedscope",
        "--rate", str(args.rate),
        "-o", str(speedscope),
        "--",
        args.python, "-c", target,
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"Output:  {out_dir}")
    print(f"Logs:    {log_path}")
    start = time.perf_counter()
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT)
    duration = time.perf_counter() - start

    declearn_version = capture_version(
        [args.python, "-c", "import importlib.metadata; print(importlib.metadata.version('declearn'))"]
    )
    pyspy_version = capture_version([args.pyspy, "--version"])
    python_version = capture_version([args.python, "--version"])

    metadata = {
        "timestamp": ts,
        "config": str(config_path),
        "tag": tag,
        "sampling_rate_hz": args.rate,
        "wall_clock_seconds": round(duration, 2),
        "return_code": result.returncode,
        "status": "success" if result.returncode == 0 else "failed",
        "python": args.python,
        "python_version": python_version,
        "declearn_version": declearn_version,
        "py_spy_version": pyspy_version,
        "speedscope_path": str(speedscope),
        "log_path": str(log_path),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== profile complete ===")
    print(f"Duration: {metadata['wall_clock_seconds']}s")
    print(f"Status:   {metadata['status']}")
    print(f"Output:   {out_dir}")
    if result.returncode != 0:
        print(f"FAILED — see {log_path} for details.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
