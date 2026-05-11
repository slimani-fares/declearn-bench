"""End-to-end smoke: 1 round secagg-masking at N=5 on master and variant.

Confirms (a) the run completes without errors on both arms,
(b) final aggregated weights / loss are equivalent (up to floating-point
reordering noise) when arms are given the same data split + seeds.
"""

import re
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path("/home/fslimani/declearn-bench")
EXP = REPO / "declearn-experiments" / "exp_12_secagg_aggregate"
FORK = REPO / "declearn-for-exp_12_secagg_aggregate"
PY = "/home/fslimani/.venvs/declearn313/bin/python"


def install(branch):
    subprocess.run(["git", "checkout", branch], cwd=FORK, check=True,
                   capture_output=True)
    subprocess.run([PY, "-m", "pip", "install", "-e", str(FORK), "-q"],
                   check=True, capture_output=True)


def run_once(branch_label, cfg_path, log_path):
    for p in EXP.glob("result_*"):
        shutil.rmtree(p, ignore_errors=True)
    cmd = [
        PY, "-c",
        f"import asyncio; from declearn.quickrun._run import quickrun; "
        f"asyncio.run(quickrun({str(cfg_path)!r}, secagg_variant='masking'))"
    ]
    start = time.perf_counter()
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, cwd=REPO, stdout=f,
                                stderr=subprocess.STDOUT)
    duration = time.perf_counter() - start
    print(f"  [{branch_label}] wall={duration:.2f}s rc={result.returncode}")
    if result.returncode:
        print("STDOUT/STDERR tail:")
        with open(log_path) as f:
            print("".join(f.readlines()[-15:]))
        raise RuntimeError(f"{branch_label} run failed")
    return duration


def parse_loss(log_path):
    text = log_path.read_text()
    m = re.search(r"Averaged loss is:\s*([0-9.eE+\-]+)", text)
    return float(m.group(1)) if m else None


def main():
    cfg = EXP / "config_secagg_n5.toml"
    runs_dir = EXP / "runs"
    runs_dir.mkdir(exist_ok=True)
    print("=== exp_12 e2e smoke: N=5 secagg-masking ===\n")

    print("--- master ---")
    install("master")
    log_m = runs_dir / "smoke_e2e_master.log"
    t_m = run_once("master", cfg, log_m)
    loss_m = parse_loss(log_m)

    print("\n--- variant ---")
    install("exp_12_secagg_aggregate_variant")
    log_v = runs_dir / "smoke_e2e_variant.log"
    t_v = run_once("variant", cfg, log_v)
    loss_v = parse_loss(log_v)

    print("\n=== resetting venv ===")
    subprocess.run([PY, "-m", "pip", "install", "-e", str(REPO / "declearn"),
                    "-q"], check=True, capture_output=True)

    print(f"\nmaster  loss={loss_m}  wall={t_m:.2f}s")
    print(f"variant loss={loss_v}  wall={t_v:.2f}s")
    if loss_m is None or loss_v is None:
        print("WARN: could not parse loss from one of the arms")
        return
    delta = abs(loss_m - loss_v)
    rel = delta / max(abs(loss_m), 1e-12)
    print(f"delta loss = {delta:.6e}  (relative {rel:.3e})")
    if rel > 1e-3:
        print(f"FAIL: relative loss delta exceeds 1e-3")
        raise SystemExit(1)
    print("PASS: e2e arms agree within 1e-3 relative loss")


if __name__ == "__main__":
    main()
