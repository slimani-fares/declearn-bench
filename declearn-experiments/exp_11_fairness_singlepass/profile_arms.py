"""exp_11 py-spy: FairGrad master vs variant, single-seed 2-round."""

import shutil
import subprocess
import time
from pathlib import Path

REPO = Path("/home/fslimani/declearn-bench")
EXP = REPO / "declearn-experiments" / "exp_11_fairness_singlepass"
FORK = REPO / "declearn-for-exp_11_fairness_singlepass"
PY = "/home/fslimani/.venvs/declearn313/bin/python"

ARMS = [
    ("master", "master"),
    ("variant", "exp_11_fairness_singlepass_variant"),
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
         "--n_shards", "3", "--scheme", "iid", "--seed", str(seed)],
        cwd=REPO, check=True, capture_output=True,
    )
    import numpy as np
    src_root = src / "data_iid"
    dst_root = src / "data_iid_fair"
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir()
    for client_dir in sorted(src_root.iterdir()):
        if not client_dir.is_dir():
            continue
        out = dst_root / client_dir.name
        out.mkdir()
        for split in ("train", "valid"):
            shutil.copy(client_dir / f"{split}_data.npy", out / f"{split}_data.npy")
            tgt = np.load(client_dir / f"{split}_target.npy")
            bin_tgt = (tgt >= 5).astype(np.int64)
            s_attr = (tgt % 2).astype(np.float32).reshape(-1, 1)
            np.save(out / f"{split}_target.npy", bin_tgt)
            np.save(out / f"{split}_s_attr.npy", s_attr)


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
    latest = sorted((out_root / arm_label).iterdir())[-1]
    print(f"  speedscope: {latest / 'pyspy_speedscope.json'}")


def main():
    cfg = EXP / "config_fairgrad.toml"
    print("=== exp_11 py-spy: FairGrad master + variant ===\n")
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
