"""exp_04 follow-up: smoke each variant at 1 round, single seed.

Confirms each variant produces an end-of-round epsilon byte-identical to
the canonical baseline (because all variants preserve accountant.step()
calls in identical order; only the epsilon-query timing differs).
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/fslimani/declearn-bench")
EXP = REPO / "declearn-experiments" / "exp_04_dp"
FORK = REPO / "declearn-for-exp_04_dp"
VENV_PY = "/home/fslimani/.venvs/declearn313/bin/python"

BRANCHES = [
    ("canonical", "master"),
    ("h1", "exp_04_dp_variant_h1"),
    ("h2_periodic", "exp_04_dp_variant_h2_periodic"),
    ("h3_precompute", "exp_04_dp_variant_h3_precompute"),
    ("h4_adaptive", "exp_04_dp_variant_h4_adaptive"),
]


def install_branch(branch):
    subprocess.run(["git", "checkout", branch], cwd=FORK, check=True,
                   capture_output=True)
    subprocess.run([VENV_PY, "-m", "pip", "install", "-e", str(FORK), "-q"],
                   cwd=REPO, check=True, capture_output=True)


def regenerate_split_seed42():
    """Regenerate seed-42 IID split + chw layout, idempotent."""
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


def run_one(label):
    """Run one quickrun with whatever fork is currently installed."""
    # Use the 1-round smoke config; create if missing
    smoke_cfg = EXP / "config_smoke_1r.toml"
    if not smoke_cfg.exists():
        text = (EXP / "config_variant.toml").read_text().replace(
            "rounds = 2", "rounds = 1"
        )
        smoke_cfg.write_text(text)
    # Wipe any prior result_* dir so quickrun doesn't pick up stale state
    for p in EXP.glob("result_*"):
        shutil.rmtree(p, ignore_errors=True)
    log_path = EXP / "runs" / f"smoke_followup_{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cmd = [
        VENV_PY, "-c",
        f"import asyncio; from declearn.quickrun._run import quickrun; "
        f"asyncio.run(quickrun({str(smoke_cfg)!r}))"
    ]
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    duration = time.perf_counter() - start
    log = log_path.read_text()
    eps = None; delta = None; acc_clients = []
    for line in log.splitlines():
        if "DP budget spent at the end of the round" in line:
            try:
                tup = line.split(": (")[-1].rstrip(")")
                e, d = tup.split(",")
                eps = float(e.strip()); delta = float(d.strip())
            except Exception:
                pass
        if "Local scalar evaluation metrics" in line and "accuracy" in line:
            try:
                acc_clients.append(float(line.split("'accuracy': '")[1].split("'")[0]))
            except Exception:
                pass
    return {
        "label": label,
        "wall_clock": round(duration, 2),
        "return_code": result.returncode,
        "final_epsilon": eps,
        "final_delta": delta,
        "client_accuracies": acc_clients,
        "mean_accuracy": (sum(acc_clients) / len(acc_clients)) if acc_clients else None,
    }


def main():
    print("=== exp_04 follow-up smoke (1 round × seed 42 × all variants) ===\n")
    regenerate_split_seed42()
    results = []
    for label, branch in BRANCHES:
        print(f"--- {label} (branch {branch}) ---")
        install_branch(branch)
        r = run_one(label)
        results.append(r)
        print(f"  wall={r['wall_clock']}s rc={r['return_code']} "
              f"eps={r['final_epsilon']} acc={r['mean_accuracy']}")
    out = EXP / "runs" / "smoke_followup_summary.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsmoke summary -> {out}")
    canon_eps = results[0]["final_epsilon"]
    print(f"\n=== epsilon equivalence (vs canonical={canon_eps}) ===")
    all_match = True
    for r in results[1:]:
        match = r["final_epsilon"] == canon_eps
        print(f"  {r['label']}: eps={r['final_epsilon']} {'OK' if match else 'MISMATCH'}")
        if not match:
            all_match = False
    print(f"\nresult: {'ALL VARIANTS BYTE-IDENTICAL' if all_match else 'MISMATCH(ES) — investigate before A/B'}")
    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
