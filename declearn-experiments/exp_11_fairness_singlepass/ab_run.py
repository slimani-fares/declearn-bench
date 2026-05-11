"""exp_11 A/B: 3 fairness algos × 2 arms × 3 seeds × 2 rounds = 18 runs.

Compares wall-clock between the canonical declearn (master arm of fork —
unified runner + canonical fairness compute) and the single-pass variant.
"""

import json
import shutil
import statistics as st
import subprocess
import time
from pathlib import Path

REPO = Path("/home/fslimani/declearn-bench")
EXP = REPO / "declearn-experiments" / "exp_11_fairness_singlepass"
FORK = REPO / "declearn-for-exp_11_fairness_singlepass"
PY = "/home/fslimani/.venvs/declearn313/bin/python"

SEEDS = [42, 43, 44]
ROUNDS = 2

ALGOS = [
    ("fairgrad", EXP / "config_fairgrad.toml"),
    ("fairbatch", EXP / "config_fairbatch.toml"),
    ("fairfed", EXP / "config_fairfed.toml"),
]
ARMS = [
    ("master", "master"),
    ("variant", "exp_11_fairness_singlepass_variant"),
]


def install(branch):
    subprocess.run(["git", "checkout", branch], cwd=FORK, check=True,
                   capture_output=True)
    subprocess.run([PY, "-m", "pip", "install", "-e", str(FORK), "-q"],
                   check=True, capture_output=True)


def regenerate_split(seed):
    """Regen data_iid_fair (3 clients, given seed)."""
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
    # Build the fairness-augmented variant.
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


def run_one(algo_label, cfg_path, arm_label, seed):
    for p in EXP.glob("result_*"):
        shutil.rmtree(p, ignore_errors=True)
    log_path = EXP / "runs" / f"ab_{algo_label}_{arm_label}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cmd = [
        PY, "-c",
        f"import asyncio; from declearn.quickrun._run import quickrun; "
        f"asyncio.run(quickrun({str(cfg_path)!r}))"
    ]
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, cwd=REPO, stdout=f,
                                stderr=subprocess.STDOUT)
    duration = time.perf_counter() - start
    log = log_path.read_text()
    accs = []
    for line in log.splitlines():
        if "Local scalar evaluation metrics" in line and "accuracy" in line:
            try:
                accs.append(float(line.split("'accuracy': '")[1].split("'")[0]))
            except Exception:
                pass
    return {
        "algo": algo_label,
        "arm": arm_label,
        "seed": seed,
        "wall_clock": round(duration, 3),
        "return_code": result.returncode,
        "accuracies": accs,
        "mean_accuracy": (sum(accs) / len(accs)) if accs else None,
    }


def main():
    print(f"=== exp_11 A/B: {len(ALGOS)} algos × {len(ARMS)} arms × "
          f"{len(SEEDS)} seeds × {ROUNDS} rounds ===\n")
    all_results = []
    for seed in SEEDS:
        print(f"\n--- seed {seed}: regenerating split ---")
        regenerate_split(seed)
        for algo_label, cfg in ALGOS:
            for arm_label, branch in ARMS:
                print(f"  {algo_label:<10} {arm_label:<8} (branch {branch}) ", end="", flush=True)
                install(branch)
                r = run_one(algo_label, cfg, arm_label, seed)
                all_results.append(r)
                print(f"wall={r['wall_clock']:>7.2f}s rc={r['return_code']} "
                      f"acc={r['mean_accuracy']}")

    out = EXP / "runs" / "ab_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nresults -> {out}")

    # Summary
    print(f"\n=== summary (n_seeds={len(SEEDS)}, rounds={ROUNDS}) ===\n")
    print(f"{'algo':<10} {'arm':<8} {'wall_mean':>10} {'wall_std':>10} "
          f"{'speedup_vs_master':>20} {'acc_mean':>10}")
    for algo_label, _ in ALGOS:
        master_walls = [r["wall_clock"] for r in all_results
                        if r["algo"] == algo_label and r["arm"] == "master"]
        master_mean = st.mean(master_walls) if master_walls else None
        for arm_label, _ in ARMS:
            rs = [r for r in all_results
                  if r["algo"] == algo_label and r["arm"] == arm_label]
            walls = [r["wall_clock"] for r in rs]
            accs = [r["mean_accuracy"] for r in rs if r["mean_accuracy"] is not None]
            wm = st.mean(walls) if walls else 0.0
            ws = st.stdev(walls) if len(walls) > 1 else 0.0
            am = st.mean(accs) if accs else 0.0
            sp = (master_mean / wm) if (master_mean and wm) else 0.0
            print(f"{algo_label:<10} {arm_label:<8} {wm:>10.2f} {ws:>10.2f} "
                  f"{sp:>19.2f}x {am:>10.4f}")

    print("\n=== resetting venv to canonical declearn ===")
    subprocess.run([PY, "-m", "pip", "install", "-e", str(REPO / "declearn"),
                    "-q"], check=True, capture_output=True)


if __name__ == "__main__":
    main()
