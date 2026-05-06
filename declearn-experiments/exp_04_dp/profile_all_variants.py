"""exp_04 follow-up: instrumented profile of every variant arm.

Runs each of the 5 arms (canonical + H1/H2/H3/H4) under py-spy at 2 rounds,
single seed (42). Produces a speedscope JSON per arm under
runs/profiles_followup/<arm>/<timestamp>/pyspy_speedscope.json so the
post-A/B hot-spot landscape can be inspected for any variant.

Single-seed because:
  - This is for *inspection* (where did the time go), not for statistics
    (which we already have from ab_followup.py at 3 seeds, no py-spy).
  - py-spy adds ~5-10% sampling overhead; the wall-clock numbers in the
    metadata.json are therefore NOT directly comparable to the A/B
    numbers. Use ab_followup_results.json for ranking; use these
    speedscope JSONs for "what's hot now".
"""

import json
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path("/home/fslimani/declearn-bench")
EXP = REPO / "declearn-experiments" / "exp_04_dp"
FORK = REPO / "declearn-for-exp_04_dp"
VENV_PY = "/home/fslimani/.venvs/declearn313/bin/python"

ARMS = [
    ("canonical", "master"),
    ("h1_deferred", "exp_04_dp_variant_h1"),
    ("h2_periodic", "exp_04_dp_variant_h2_periodic"),
    ("h3_precompute", "exp_04_dp_variant_h3_precompute"),
    ("h4_adaptive", "exp_04_dp_variant_h4_adaptive"),
]


def install_branch(branch):
    subprocess.run(["git", "checkout", branch], cwd=FORK, check=True,
                   capture_output=True)
    subprocess.run([VENV_PY, "-m", "pip", "install", "-e", str(FORK), "-q"],
                   cwd=REPO, check=True, capture_output=True)


def regenerate_seed42():
    src = REPO / "examples" / "mnist_quickrun"
    for d in src.glob("data_iid*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    subprocess.run(
        ["declearn-split", "--folder", "examples/mnist_quickrun",
         "--n_shards", "2", "--scheme", "iid", "--seed", "42"],
        cwd=REPO, check=True, capture_output=True,
    )
    import numpy as np
    src_d = src / "data_iid"
    dst_d = src / "data_iid_chw"
    dst_d.mkdir(exist_ok=True)
    for client in sorted(src_d.iterdir()):
        if not client.is_dir():
            continue
        out = dst_d / client.name
        out.mkdir(exist_ok=True)
        for split in ("train", "valid"):
            d = np.load(client / f"{split}_data.npy")
            t = np.load(client / f"{split}_target.npy")
            np.save(out / f"{split}_data.npy", d.reshape(d.shape[0], 1, 28, 28))
            np.save(out / f"{split}_target.npy", t.astype(np.int64))


def profile_one(arm_label):
    """Wrap run_profile.py around the currently-installed declearn fork."""
    cfg = EXP / "config_variant.toml"  # rounds=2
    out_dir = EXP / "runs" / "profiles_followup"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Wipe stale checkpoints
    for p in EXP.glob("result_*"):
        shutil.rmtree(p, ignore_errors=True)
    start = time.perf_counter()
    cmd = [
        VENV_PY, "profiling/run_profile.py",
        "--config", str(cfg),
        "--tag", arm_label,
        "--out", str(out_dir),
    ]
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    duration = time.perf_counter() - start
    return {
        "arm": arm_label,
        "wall_clock_pyspy": round(duration, 2),
        "return_code": result.returncode,
        "stdout_tail": result.stdout.splitlines()[-3:] if result.stdout else [],
    }


def main():
    print("=== exp_04 follow-up: py-spy instrumentation of all 5 arms ===\n")
    print("(NB: py-spy adds ~5-10% sampling overhead; wall-clock here is")
    print(" inflated vs ab_followup_results.json. Use these JSONs for")
    print(" hot-spot inspection, not for the speedup ranking.)\n")
    regenerate_seed42()
    summary = []
    for arm, branch in ARMS:
        print(f"--- {arm} (branch {branch}) ---")
        install_branch(branch)
        r = profile_one(arm)
        summary.append(r)
        for line in r["stdout_tail"]:
            print(f"  {line}")
        print(f"  pyspy-instrumented wall: {r['wall_clock_pyspy']}s")
    out = EXP / "runs" / "profiles_followup_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {out}")
    print("\nspeedscope JSONs (drop into https://www.speedscope.app to view):")
    for arm, _ in ARMS:
        latest = sorted((EXP / "runs" / "profiles_followup" / arm).iterdir())
        if latest:
            print(f"  {arm}: {latest[-1] / 'pyspy_speedscope.json'}")


if __name__ == "__main__":
    main()
