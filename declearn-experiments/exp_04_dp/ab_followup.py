"""exp_04 follow-up A/B: 5 arms × 3 seeds × 2 rounds.

Arms: canonical (master), H1 deferred, H2 periodic K=10, H3 precompute,
H4 adaptive.
Compares: wall-clock, end-of-round epsilon trajectory, mean validation
accuracy.
Privacy invariant: all 5 arms must produce byte-identical end-of-round
epsilon (proven at smoke; verified again here per seed). Wall-clock
delta is the only meaningful comparison metric.
"""

import json
import shutil
import subprocess
import statistics as st
import time
from pathlib import Path

REPO = Path("/home/fslimani/declearn-bench")
EXP = REPO / "declearn-experiments" / "exp_04_dp"
FORK = REPO / "declearn-for-exp_04_dp"
VENV_PY = "/home/fslimani/.venvs/declearn313/bin/python"

SEEDS = [42, 43, 44]
ROUNDS = 2

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


def regenerate_split(seed):
    src = REPO / "examples" / "mnist_quickrun"
    for d in src.glob("data_iid*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    subprocess.run(
        ["declearn-split", "--folder", "examples/mnist_quickrun",
         "--n_shards", "2", "--scheme", "iid", "--seed", str(seed)],
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


def run_one(label, seed):
    """Run 2-round DP quickrun with whatever fork is currently installed."""
    cfg = EXP / "config_variant.toml"  # already rounds=2
    for p in EXP.glob("result_*"):
        shutil.rmtree(p, ignore_errors=True)
    log_path = EXP / "runs" / f"ab_followup_{label}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cmd = [
        VENV_PY, "-c",
        f"import asyncio; from declearn.quickrun._run import quickrun; "
        f"asyncio.run(quickrun({str(cfg)!r}))"
    ]
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    duration = time.perf_counter() - start
    log = log_path.read_text()
    eps_per_round = []
    acc_clients = []
    for line in log.splitlines():
        if "DP budget spent at the end of the round" in line:
            try:
                tup = line.split(": (")[-1].rstrip(")")
                e, d = tup.split(",")
                eps_per_round.append(float(e.strip()))
            except Exception:
                pass
        if "Local scalar evaluation metrics" in line and "accuracy" in line:
            try:
                acc_clients.append(float(line.split("'accuracy': '")[1].split("'")[0]))
            except Exception:
                pass
    return {
        "label": label,
        "seed": seed,
        "wall_clock": round(duration, 3),
        "return_code": result.returncode,
        "eps_per_round": eps_per_round,
        "final_epsilon": eps_per_round[-1] if eps_per_round else None,
        "client_accuracies": acc_clients,
        "mean_accuracy": (sum(acc_clients) / len(acc_clients)) if acc_clients else None,
    }


def main():
    print(f"=== exp_04 follow-up A/B: 5 arms × {len(SEEDS)} seeds × {ROUNDS} rounds ===\n")

    # Iterate by SEED outer, ARM inner so each seed's arms run back-to-back
    # under the same data split (max comparability).
    all_results = {label: [] for label, _ in ARMS}
    for seed in SEEDS:
        print(f"\n=== seed {seed}: regenerating data split ===")
        regenerate_split(seed)
        for label, branch in ARMS:
            print(f"--- seed {seed} | {label} (branch {branch}) ---")
            install_branch(branch)
            r = run_one(label, seed)
            all_results[label].append(r)
            print(f"  wall={r['wall_clock']}s rc={r['return_code']} "
                  f"eps_per_round={r['eps_per_round']} acc={r['mean_accuracy']}")

    out = EXP / "runs" / "ab_followup_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nresults -> {out}")

    # Summary table
    def stats(vals):
        clean = [v for v in vals if v is not None]
        if not clean:
            return (None, None)
        return (st.mean(clean), st.stdev(clean) if len(clean) > 1 else 0.0)

    print(f"\n=== A/B summary (n_seeds={len(SEEDS)}, rounds={ROUNDS}, n_clients=2) ===\n")
    print(f"{'arm':<15} {'wall_mean':>10} {'wall_std':>10} {'speedup':>10} "
          f"{'final_eps':>20} {'acc_mean':>10}")
    canon_wall, _ = stats([r["wall_clock"] for r in all_results["canonical"]])
    for label, _ in ARMS:
        rs = all_results[label]
        wm, ws = stats([r["wall_clock"] for r in rs])
        epsilons = [r["final_epsilon"] for r in rs if r["final_epsilon"] is not None]
        eps_str = f"{epsilons[0]:.6f}" if epsilons else "(missing)"
        eps_match = "OK" if all(abs(e - epsilons[0]) < 1e-12 for e in epsilons) else "MISMATCH"
        am, _ = stats([r["mean_accuracy"] for r in rs])
        speedup = (canon_wall / wm) if (canon_wall and wm) else 0.0
        print(f"{label:<15} {wm:>10.2f} {ws:>10.2f} {speedup:>10.2f}x "
              f"{eps_str:>20} {(am or 0):>10.4f}")
        if eps_match != "OK":
            print(f"  WARNING: epsilon varies across seeds for {label}: {epsilons}")

    # Cross-arm epsilon equivalence (same seed, all arms should produce the
    # same end-of-round epsilon trajectory)
    print(f"\n=== cross-arm epsilon equivalence (per seed) ===")
    for i, seed in enumerate(SEEDS):
        canon_eps = all_results["canonical"][i]["eps_per_round"]
        all_match = True
        for label, _ in ARMS[1:]:
            ep = all_results[label][i]["eps_per_round"]
            if ep != canon_eps:
                print(f"  seed {seed}: {label} eps_per_round={ep} != "
                      f"canonical {canon_eps}  MISMATCH")
                all_match = False
        if all_match:
            print(f"  seed {seed}: all arms match canonical {canon_eps}")


if __name__ == "__main__":
    main()
