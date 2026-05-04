"""exp_04 A/B: baseline vs variant DP across multiple seeds, 1 round, 2 clients."""

import json
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path("/home/fslimani/declearn-bench")
EXP = REPO / "declearn-experiments" / "exp_04_dp"
FORK = REPO / "declearn-for-exp_04_dp"
VENV_PY = "/home/fslimani/.venvs/declearn313/bin/python"
SEEDS = [42, 43, 44]   # default 3; bump to [42..46] if results inconclusive
ROUNDS = 1


def install_branch(branch):
    subprocess.run(["git", "checkout", branch], cwd=FORK, check=True,
                   capture_output=True)
    subprocess.run([VENV_PY, "-m", "pip", "install", "-e", str(FORK), "-q"],
                   cwd=REPO, check=True, capture_output=True)


def regenerate_split(seed):
    """Regenerate the data split with this seed; rebuild data_iid_chw."""
    src = REPO / "examples" / "mnist_quickrun"
    # Delete existing splits
    for d in src.glob("data_iid*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    subprocess.run(
        ["declearn-split", "--folder", "examples/mnist_quickrun",
         "--n_shards", "2", "--scheme", "iid", "--seed", str(seed)],
        cwd=REPO, check=True, capture_output=True,
    )
    # Rebuild chw layout from the just-created data_iid
    import numpy as np
    src_d = src / "data_iid"
    dst_d = src / "data_iid_chw"
    dst_d.mkdir(exist_ok=True)
    for client in sorted(src_d.iterdir()):
        if not client.is_dir():
            continue
        out = dst_d / client.name; out.mkdir(exist_ok=True)
        for split in ("train", "valid"):
            d = np.load(client / f"{split}_data.npy")
            t = np.load(client / f"{split}_target.npy")
            np.save(out / f"{split}_data.npy", d.reshape(d.shape[0], 1, 28, 28))
            np.save(out / f"{split}_target.npy", t.astype(np.int64))


def run_dp(config_path, label, seed):
    log_path = EXP / "runs" / f"ab_{label}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Wipe any previous result_* dir for the experience
    for p in EXP.glob("result_*"):
        shutil.rmtree(p, ignore_errors=True)
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
        "label": label, "seed": seed,
        "wall_clock": round(duration, 2),
        "return_code": result.returncode,
        "final_epsilon": eps,
        "final_delta": delta,
        "client_accuracies": acc_clients,
        "mean_accuracy": (sum(acc_clients) / len(acc_clients)) if acc_clients else None,
    }


def main():
    # Use smoke 1-round config
    cfg = EXP / "config_smoke_1r.toml"
    if not cfg.exists():
        text = (EXP / "config_variant.toml").read_text().replace("rounds = 2", "rounds = 1")
        cfg.write_text(text)

    results = {"baseline": [], "variant": []}

    for seed in SEEDS:
        print(f"\n=== seed {seed}: regenerating split ===")
        regenerate_split(seed)

        print(f"--- baseline @ seed {seed} ---")
        install_branch("master")
        r = run_dp(cfg, "baseline", seed)
        results["baseline"].append(r)
        print(f"  wall={r['wall_clock']}s eps={r['final_epsilon']} acc={r['mean_accuracy']}")

        print(f"--- variant @ seed {seed} ---")
        install_branch("exp_04_dp_variant_h1")
        r = run_dp(cfg, "variant", seed)
        results["variant"].append(r)
        print(f"  wall={r['wall_clock']}s eps={r['final_epsilon']} acc={r['mean_accuracy']}")

    out_path = EXP / "runs" / "ab_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nA/B results → {out_path}")

    # Stats
    import statistics as st
    def stats(arr, key):
        vals = [r[key] for r in arr if r[key] is not None]
        return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)
    bw = stats(results["baseline"], "wall_clock")
    vw = stats(results["variant"], "wall_clock")
    ba = stats(results["baseline"], "mean_accuracy")
    va = stats(results["variant"], "mean_accuracy")
    be = stats(results["baseline"], "final_epsilon")
    ve = stats(results["variant"], "final_epsilon")
    print(f"\n=== A/B summary (n_seeds={len(SEEDS)}, rounds={ROUNDS}, n_clients=2) ===")
    print(f"wall-clock: baseline {bw[0]:.2f}s ± {bw[1]:.2f}   variant {vw[0]:.2f}s ± {vw[1]:.2f}   "
          f"speedup: {bw[0]/vw[0]:.2f}x")
    print(f"epsilon:    baseline {be[0]:.6f}             variant {ve[0]:.6f}             "
          f"delta: {ve[0]-be[0]:+.6f}")
    print(f"accuracy:   baseline {ba[0]:.4f} ± {ba[1]:.4f}  variant {va[0]:.4f} ± {va[1]:.4f}  "
          f"delta: {va[0]-ba[0]:+.4f}")


if __name__ == "__main__":
    main()
