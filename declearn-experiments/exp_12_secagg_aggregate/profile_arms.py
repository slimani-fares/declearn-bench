"""exp_12 py-spy: secagg-masking N=20 master vs variant."""

import os
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path("/home/fslimani/declearn-bench")
EXP = REPO / "declearn-experiments" / "exp_12_secagg_aggregate"
FORK = REPO / "declearn-for-exp_12_secagg_aggregate"
PY = "/home/fslimani/.venvs/declearn313/bin/python"
PYSPY = "/home/fslimani/.venvs/declearn313/bin/py-spy"

ARMS = [
    ("master", "master"),
    ("variant", "exp_12_secagg_aggregate_variant"),
]


def install(branch):
    subprocess.run(["git", "checkout", branch], cwd=FORK, check=True,
                   capture_output=True)
    subprocess.run([PY, "-m", "pip", "install", "-e", str(FORK), "-q"],
                   check=True, capture_output=True)


def profile_arm(arm_label):
    cfg = EXP / "config_secagg_n20.toml"
    out_root = EXP / "runs" / "profiles_n20"
    out_root.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    arm_dir = out_root / arm_label / ts
    arm_dir.mkdir(parents=True, exist_ok=True)
    speedscope = arm_dir / "pyspy_speedscope.json"
    for p in EXP.glob("result_*"):
        shutil.rmtree(p, ignore_errors=True)
    target = (
        "import asyncio; "
        "from declearn.quickrun._run import quickrun; "
        f"asyncio.run(quickrun({str(cfg)!r}, secagg_variant='masking'))"
    )
    cmd = [
        PYSPY, "record",
        "--format", "speedscope",
        "--rate", "100",
        "-o", str(speedscope),
        "--",
        PY, "-c", target,
    ]
    start = time.perf_counter()
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    dur = time.perf_counter() - start
    print(f"[{arm_label}] wall={dur:.2f}s rc={r.returncode}")
    if r.returncode:
        print("STDERR tail:", r.stderr.splitlines()[-5:] if r.stderr else "")
    print(f"  speedscope: {speedscope}")


def main():
    print("=== exp_12 py-spy: secagg-masking N=20 ===\n")
    for arm_label, branch in ARMS:
        print(f"--- {arm_label} (branch {branch}) ---")
        install(branch)
        profile_arm(arm_label)
    print("\n=== resetting venv to canonical declearn ===")
    subprocess.run([PY, "-m", "pip", "install", "-e", str(REPO / "declearn"),
                    "-q"], check=True, capture_output=True)


if __name__ == "__main__":
    main()
