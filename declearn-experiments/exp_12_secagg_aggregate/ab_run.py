"""exp_12 A/B: secagg-masking, 3 N-values × 2 arms × 3 seeds × 1 round.

Hypothesis: aggregate-side cost is O(N · L) per round; speedup from
the variant should grow with N. At N=5 the encrypt-side dominates;
at N=20 the aggregate-side should be visible.

Note: the seed only changes torch RNG (reseeded per subprocess); the
data split is fixed per N. We use seed only as an A/B repeat counter
to characterize noise — secagg is deterministic given identity keys.
"""

import json
import re
import shutil
import statistics as st
import subprocess
import time
from pathlib import Path

REPO = Path("/home/fslimani/declearn-bench")
EXP = REPO / "declearn-experiments" / "exp_12_secagg_aggregate"
FORK = REPO / "declearn-for-exp_12_secagg_aggregate"
PY = "/home/fslimani/.venvs/declearn313/bin/python"

N_VALUES = [5, 10, 20]
SEEDS = [42, 43, 44]
ROUNDS = 1

ARMS = [
    ("master", "master"),
    ("variant", "exp_12_secagg_aggregate_variant"),
]


def install(branch):
    subprocess.run(["git", "checkout", branch], cwd=FORK, check=True,
                   capture_output=True)
    subprocess.run([PY, "-m", "pip", "install", "-e", str(FORK), "-q"],
                   check=True, capture_output=True)


def run_one(n, arm_label, seed):
    cfg = EXP / f"config_secagg_n{n}.toml"
    for p in EXP.glob("result_*"):
        shutil.rmtree(p, ignore_errors=True)
    log_path = EXP / "runs" / f"ab_n{n}_{arm_label}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cmd = [
        PY, "-c",
        f"import asyncio; from declearn.quickrun._run import quickrun; "
        f"asyncio.run(quickrun({str(cfg)!r}, secagg_variant='masking'))"
    ]
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, cwd=REPO, stdout=f,
                                stderr=subprocess.STDOUT)
    duration = time.perf_counter() - start
    text = log_path.read_text()
    loss_m = re.search(r"Averaged loss is:\s*([0-9.eE+\-]+)", text)
    return {
        "n": n, "arm": arm_label, "seed": seed,
        "wall_clock": round(duration, 3),
        "return_code": result.returncode,
        "loss": float(loss_m.group(1)) if loss_m else None,
    }


def main():
    print(f"=== exp_12 A/B: N={N_VALUES} × {len(ARMS)} arms × "
          f"{len(SEEDS)} seeds × {ROUNDS} rounds ===\n")
    all_results = []
    for n in N_VALUES:
        for seed in SEEDS:
            for arm_label, branch in ARMS:
                print(f"  N={n:>2} {arm_label:<8} seed={seed} ", end="", flush=True)
                install(branch)
                r = run_one(n, arm_label, seed)
                all_results.append(r)
                print(f"wall={r['wall_clock']:>7.2f}s rc={r['return_code']} "
                      f"loss={r['loss']}")

    out = EXP / "runs" / "ab_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nresults -> {out}")

    # Summary
    print(f"\n=== summary ===\n")
    print(f"{'N':>3} {'arm':<8} {'wall_mean':>10} {'wall_std':>10} "
          f"{'speedup':>10} {'loss_mean':>14}")
    for n in N_VALUES:
        master_walls = [r["wall_clock"] for r in all_results
                        if r["n"] == n and r["arm"] == "master"]
        master_mean = st.mean(master_walls) if master_walls else None
        for arm_label, _ in ARMS:
            rs = [r for r in all_results
                  if r["n"] == n and r["arm"] == arm_label]
            walls = [r["wall_clock"] for r in rs]
            losses = [r["loss"] for r in rs if r["loss"] is not None]
            wm = st.mean(walls) if walls else 0.0
            ws = st.stdev(walls) if len(walls) > 1 else 0.0
            lm = st.mean(losses) if losses else 0.0
            sp = (master_mean / wm) if (master_mean and wm) else 0.0
            print(f"{n:>3} {arm_label:<8} {wm:>10.2f} {ws:>10.2f} "
                  f"{sp:>9.2f}x {lm:>14.6f}")

    print("\n=== resetting venv to canonical declearn ===")
    subprocess.run([PY, "-m", "pip", "install", "-e", str(REPO / "declearn"),
                    "-q"], check=True, capture_output=True)


if __name__ == "__main__":
    main()
