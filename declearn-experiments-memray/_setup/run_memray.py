"""Shared driver: run a python target under memray, generate flamegraph +
summary text, then drop the .bin to save disk.

Usage:
    python run_memray.py <out_dir> <tag> <python_target_string>

The target is the same kind of `-c` payload used in our other drivers:
    "import asyncio; from declearn.quickrun._run import quickrun; "
    "asyncio.run(quickrun('/abs/path/cfg.toml'))"

Outputs into <out_dir>/<tag>/<timestamp>/:
    flamegraph.html   visual heap-allocation flame, browser-openable
    summary.txt       memray summary + memray stats text reports
    metadata.json     run timing, target config, etc.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

VENV_BIN = Path("/home/fslimani/.venvs/declearn313/bin")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--target", required=True,
                   help="Python -c payload to run under memray.")
    p.add_argument("--cwd", default="/home/fslimani/declearn-bench")
    p.add_argument("--keep-bin", action="store_true",
                   help="Keep the raw .bin (otherwise deleted after report).")
    args = p.parse_args()

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.out) / args.tag / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    bin_path = run_dir / "memray.bin"
    flame_path = run_dir / "flamegraph.html"
    summary_path = run_dir / "summary.txt"
    meta_path = run_dir / "metadata.json"

    memray = str(VENV_BIN / "memray")
    py = str(VENV_BIN / "python")

    print(f"=== memray run ===")
    print(f"tag:    {args.tag}")
    print(f"output: {run_dir}")
    print(f"target: {args.target[:120]}{'...' if len(args.target) > 120 else ''}")

    # Step 1: run target under memray. `memray run -c "..."` interprets the
    # next arg as a python command; we pass the target string this way and
    # let memray handle its own python invocation.
    cmd = [
        memray, "run",
        "-o", str(bin_path),
        "--quiet",
        "-c", args.target,
    ]
    start = time.perf_counter()
    r = subprocess.run(cmd, cwd=args.cwd, capture_output=True, text=True)
    duration = time.perf_counter() - start
    if r.returncode:
        print(f"FAIL: memray run rc={r.returncode}")
        print("STDOUT tail:", "\n".join(r.stdout.splitlines()[-15:]))
        print("STDERR tail:", "\n".join(r.stderr.splitlines()[-15:]))
        sys.exit(1)
    bin_size_mb = bin_path.stat().st_size / 1e6
    print(f"  run wall-clock: {duration:.2f}s  bin size: {bin_size_mb:.1f} MB")

    # Step 2: derivatives. Generate flamegraph + summary + stats text.
    print("  generating flamegraph + summary…")
    subprocess.run(
        [memray, "flamegraph", "--output", str(flame_path), "--force",
         str(bin_path)],
        check=True, capture_output=True, text=True,
    )
    # `memray summary` is a top-N table by total allocated bytes.
    with open(summary_path, "w") as f:
        f.write(f"=== memray summary ({args.tag} @ {ts}) ===\n\n")
        f.write("--- memray stats ---\n")
        s = subprocess.run(
            [memray, "stats", str(bin_path)],
            capture_output=True, text=True,
        )
        f.write(s.stdout)
        f.write(s.stderr)
        f.write("\n\n--- memray summary (top-25 by own memory at peak) ---\n")
        s = subprocess.run(
            [memray, "summary", "-r", "25", str(bin_path)],
            capture_output=True, text=True,
            timeout=60,
        )
        f.write(s.stdout)
        f.write(s.stderr)
        # NOTE: `memray tree` is an interactive TUI when stdin is a TTY and
        # also hangs under capture_output. Do NOT call it from this driver.

    # Step 3: metadata + cleanup.
    meta = {
        "tag": args.tag,
        "timestamp": ts,
        "wall_clock_s": round(duration, 3),
        "bin_size_mb": round(bin_size_mb, 2),
        "target": args.target,
        "cwd": args.cwd,
        "memray_version": subprocess.run(
            [memray, "--version"], capture_output=True, text=True,
        ).stdout.strip(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    if not args.keep_bin:
        bin_path.unlink()
        print(f"  .bin deleted (saved {bin_size_mb:.1f} MB)")
    print(f"  flamegraph: {flame_path}")
    print(f"  summary:    {summary_path}")
    print(f"  metadata:   {meta_path}")


if __name__ == "__main__":
    main()
