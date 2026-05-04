"""exp_04 smoke test: variant vs baseline DP-SGD, 1 round, single seed.

Compares end-of-round (epsilon, delta, final_accuracy, wall_clock) between
the canonical declearn-for-exp_04_dp baseline branch and the exp_04_dp_variant_h1
branch. The variant defers per-step get_epsilon to round boundaries.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/fslimani/declearn-bench")
EXP = REPO / "declearn-experiments" / "exp_04_dp"
FORK = REPO / "declearn-for-exp_04_dp"
VENV_PY = "/home/fslimani/.venvs/declearn313/bin/python"


def install_branch(branch):
    """Check out a branch in the fork and reinstall it editable."""
    subprocess.run(["git", "checkout", branch], cwd=FORK, check=True,
                   capture_output=True)
    subprocess.run([VENV_PY, "-m", "pip", "install", "-e", str(FORK), "-q"],
                   cwd=REPO, check=True, capture_output=True)


def run_dp(config_path, label):
    """Run a single DP quickrun, capture wall-clock and final epsilon/acc from log."""
    log_path = EXP / "runs" / f"smoke_{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cmd = [
        VENV_PY, "-c",
        f"import asyncio; from declearn.quickrun._run import quickrun; "
        f"asyncio.run(quickrun({str(config_path)!r}))"
    ]
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    duration = time.perf_counter() - start
    log = log_path.read_text()
    # Extract final budget line: "Local DP budget spent at the end of the round: (eps, delta)"
    eps = None; delta = None
    for line in log.splitlines():
        if "DP budget spent at the end of the round" in line:
            # parse last occurrence
            try:
                tup = line.split(": (")[-1].rstrip(")")
                e, d = tup.split(",")
                eps = float(e.strip()); delta = float(d.strip())
            except Exception:
                pass
    # Extract final accuracy on each client (max round)
    acc = None
    for line in reversed(log.splitlines()):
        if "Local scalar evaluation metrics" in line and "accuracy" in line:
            try:
                acc = float(line.split("'accuracy': '")[1].split("'")[0])
                break
            except Exception:
                pass
    return {
        "label": label,
        "branch": label,
        "wall_clock": round(duration, 2),
        "return_code": result.returncode,
        "final_epsilon": eps,
        "final_delta": delta,
        "final_accuracy_one_client": acc,
        "log": str(log_path),
    }


def main():
    # Use a 1-round config for smoke speed (variant TOML has rounds=2; modify in temp).
    src = EXP / "config_variant.toml"
    smoke_cfg = EXP / "config_smoke_1r.toml"
    text = src.read_text().replace("rounds = 2", "rounds = 1")
    smoke_cfg.write_text(text)

    # Clear the result_* dir from previous runs (prevents quickrun checkpoint reuse).
    for p in (EXP / "config_baseline.toml").parent.glob("result_*"):
        shutil.rmtree(p, ignore_errors=True)

    print(f"=== smoke test config: {smoke_cfg} ===")

    # Run baseline
    print("\n--- installing baseline branch ---")
    install_branch("master")
    print("running baseline...")
    res_base = run_dp(smoke_cfg, "baseline")
    print(json.dumps(res_base, indent=2))
    # Clear checkpoints between runs
    for p in EXP.glob("result_*"):
        shutil.rmtree(p, ignore_errors=True)

    # Run variant
    print("\n--- installing variant branch ---")
    install_branch("exp_04_dp_variant_h1")
    print("running variant...")
    res_var = run_dp(smoke_cfg, "variant")
    print(json.dumps(res_var, indent=2))

    # Compare
    print("\n=== smoke test summary ===")
    print(f"baseline wall: {res_base['wall_clock']}s   variant wall: {res_var['wall_clock']}s   "
          f"speedup: {res_base['wall_clock']/res_var['wall_clock']:.2f}x")
    print(f"baseline eps: {res_base['final_epsilon']}   variant eps: {res_var['final_epsilon']}   "
          f"delta: {(res_var['final_epsilon'] or 0) - (res_base['final_epsilon'] or 0):+.6f}")
    print(f"baseline acc: {res_base['final_accuracy_one_client']}   "
          f"variant acc: {res_var['final_accuracy_one_client']}   "
          f"delta: {(res_var['final_accuracy_one_client'] or 0) - (res_base['final_accuracy_one_client'] or 0):+.4f}")

    # Save summary
    summary_path = EXP / "runs" / "smoke_summary.json"
    summary_path.write_text(json.dumps({
        "baseline": res_base, "variant": res_var,
    }, indent=2))
    print(f"\nsmoke summary written to {summary_path}")


if __name__ == "__main__":
    main()
