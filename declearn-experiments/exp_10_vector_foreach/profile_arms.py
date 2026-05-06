"""exp_10 py-spy inspection: profile lasso (the heaviest) on master and variant.

Single-seed (42) per arm. Lasso chosen because it triggers BOTH paths under
the patch (apply_func(torch.sign) and _apply_operation(torch.add/...)).
Outputs speedscope JSONs for hot-spot inspection on https://www.speedscope.app.
"""

import shutil
import subprocess
import time
from pathlib import Path

REPO = Path("/home/fslimani/declearn-bench")
EXP = REPO / "declearn-experiments" / "exp_10_vector_foreach"
FORK = REPO / "declearn-for-exp_10_vector_foreach"
PY = "/home/fslimani/.venvs/declearn313/bin/python"

ARMS = [
    ("master", "master"),
    ("variant", "exp_10_vector_foreach_variant"),
]


def install(branch):
    subprocess.run(["git", "checkout", branch], cwd=FORK, check=True,
                   capture_output=True)
    subprocess.run([PY, "-m", "pip", "install", "-e", str(FORK), "-q"],
                   check=True, capture_output=True)


def regenerate_split(seed=42):
    src = REPO / "examples" / "mnist_quickrun"
    for d in src.glob("data_iid*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    subprocess.run(
        ["/home/fslimani/.venvs/declearn313/bin/declearn-split",
         "--folder", "examples/mnist_quickrun",
         "--n_shards", "2", "--scheme", "iid", "--seed", str(seed)],
        cwd=REPO, check=True, capture_output=True,
    )


def profile_arm(arm_label, cfg_path):
    out_root = EXP / "runs" / "profiles"
    out_root.mkdir(parents=True, exist_ok=True)
    for p in EXP.glob("result_*"):
        shutil.rmtree(p, ignore_errors=True)
    cmd = [
        PY, str(REPO / "profiling" / "run_profile.py"),
        "--config", str(cfg_path),
        "--tag", arm_label,
        "--out", str(out_root),
    ]
    start = time.perf_counter()
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    dur = time.perf_counter() - start
    print(f"[{arm_label}] wall={dur:.2f}s rc={r.returncode}")
    if r.returncode:
        print("STDERR tail:", r.stderr.splitlines()[-5:])
    # Print path of the latest dir
    latest = sorted((out_root / arm_label).iterdir())[-1]
    print(f"  speedscope: {latest / 'pyspy_speedscope.json'}")


def main():
    cfg = EXP / "config_lasso.toml"
    print("=== exp_10 py-spy: lasso on master + variant ===\n")
    regenerate_split(42)
    for arm_label, branch in ARMS:
        print(f"--- {arm_label} (branch {branch}) ---")
        install(branch)
        profile_arm(arm_label, cfg)

    print("\n=== resetting venv to canonical declearn ===")
    subprocess.run([PY, "-m", "pip", "install", "-e", str(REPO / "declearn"),
                    "-q"], check=True, capture_output=True)


if __name__ == "__main__":
    main()
