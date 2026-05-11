"""exp_11 smoke: per-group fairness metrics byte-equivalent vs canonical.

Strategy:
  - Install master arm of fork (canonical fairness behaviour).
  - Run a minimal FairGrad config that exercises the fairness path
    (1 round, 3 clients, fixed seed via re-generated data).
  - Parse per-client per-group `accuracy` values from the client logs
    (declearn logs them via `Local fairness measures: ...`).
  - Reinstall variant arm and repeat.
  - Compare per-(client, round, group) accuracies; assert max abs delta
    < 1e-6 (tolerance leaves headroom for float reordering, which is
    the only way the patch could differ from canonical).
"""

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/fslimani/declearn-bench")
EXP = REPO / "declearn-experiments" / "exp_11_fairness_singlepass"
FORK = REPO / "declearn-for-exp_11_fairness_singlepass"
PY = "/home/fslimani/.venvs/declearn313/bin/python"


def install(branch):
    subprocess.run(["git", "checkout", branch], cwd=FORK, check=True,
                   capture_output=True)
    subprocess.run([PY, "-m", "pip", "install", "-e", str(FORK), "-q"],
                   check=True, capture_output=True)


def run_one(branch_label, cfg_path, log_path):
    for p in EXP.glob("result_*"):
        shutil.rmtree(p, ignore_errors=True)
    cmd = [
        PY, "-c",
        f"import asyncio; from declearn.quickrun._run import quickrun; "
        f"asyncio.run(quickrun({str(cfg_path)!r}))"
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


def parse_groupwise_accuracy(log_path):
    """Return dict {(round_i, client_label): {group: accuracy}}.

    The fairness controller logs "Local fairness measures" with a
    nested dict of {metric_name: {group_tuple: value}}. We pull the
    accuracy entries.
    """
    out = {}
    text = log_path.read_text()
    # Per-round grouped log lines.
    pattern = re.compile(
        r"(.*?):declearn\.client\..*?:INFO:\s*Local fairness measures:\s*(\{.*?\})\s*$",
        re.MULTILINE,
    )
    fairness_round = 0
    cur_client = None
    for line in text.splitlines():
        m = re.search(r"declearn\.client\.(\d+)", line)
        if m:
            cur_client = int(m.group(1))
        if "Initiating fairness-enforcing round" in line:
            mr = re.search(r"round\s+(\d+)", line)
            if mr:
                fairness_round = int(mr.group(1))
        if "Local fairness measures" in line:
            i = line.find("{")
            if i >= 0:
                payload = line[i:]
                # Convert tuple keys to strings (literal_eval handles them).
                try:
                    import ast
                    parsed = ast.literal_eval(payload)
                except Exception:
                    continue
                key = (fairness_round, cur_client)
                # We want the accuracy sub-dict.
                if "accuracy" in parsed:
                    out[key] = {
                        str(g): float(v)
                        for g, v in parsed["accuracy"].items()
                    }
    return out


def main():
    cfg = EXP / "config_fairgrad.toml"
    runs_dir = EXP / "runs"
    runs_dir.mkdir(exist_ok=True)
    print("=== exp_11 smoke: FairGrad master vs variant ===\n")

    print("--- master ---")
    install("master")
    log_m = runs_dir / "smoke_master.log"
    run_one("master", cfg, log_m)
    metrics_m = parse_groupwise_accuracy(log_m)

    print("\n--- variant ---")
    install("exp_11_fairness_singlepass_variant")
    log_v = runs_dir / "smoke_variant.log"
    run_one("variant", cfg, log_v)
    metrics_v = parse_groupwise_accuracy(log_v)

    # Reset venv
    print("\n=== resetting venv ===")
    subprocess.run([PY, "-m", "pip", "install", "-e",
                    str(REPO / "declearn"), "-q"],
                   check=True, capture_output=True)

    print(f"\n--- comparing per-group accuracies ---")
    print(f"master: {len(metrics_m)} (round, client) entries")
    print(f"variant: {len(metrics_v)} (round, client) entries")
    if metrics_m.keys() != metrics_v.keys():
        only_m = set(metrics_m) - set(metrics_v)
        only_v = set(metrics_v) - set(metrics_m)
        print(f"key-set mismatch: only in master {only_m}, only in variant {only_v}")
        sys.exit(1)

    failures = []
    rows = []
    for key in sorted(metrics_m):
        m_grp = metrics_m[key]
        v_grp = metrics_v[key]
        if set(m_grp) != set(v_grp):
            failures.append((key, "group set mismatch"))
            continue
        for g in m_grp:
            d = abs(m_grp[g] - v_grp[g])
            rows.append((key, g, m_grp[g], v_grp[g], d))
            if d > 1e-6:
                failures.append((key, f"group {g}: master={m_grp[g]} variant={v_grp[g]} delta={d}"))

    print(f"\n{'(round, client)':<20} {'group':<15} {'master':>12} {'variant':>12} {'abs_diff':>14}")
    for key, g, mv, vv, d in rows:
        print(f"{str(key):<20} {g:<15} {mv:>12.8f} {vv:>12.8f} {d:>14.3e}")

    out = runs_dir / "smoke_compare.json"
    out.write_text(json.dumps({
        "master": {str(k): v for k, v in metrics_m.items()},
        "variant": {str(k): v for k, v in metrics_v.items()},
        "max_abs_diff": max((d for *_, d in rows), default=0.0),
        "failures": failures,
    }, indent=2, default=str))
    print(f"\n→ {out}")

    if failures:
        print(f"\nFAIL: {len(failures)} mismatches above 1e-6")
        for k, msg in failures:
            print(f"  {k}: {msg}")
        sys.exit(1)
    print(f"\nPASS: all {len(rows)} per-group entries match (max abs diff "
          f"{max((d for *_, d in rows), default=0.0):.3e})")


if __name__ == "__main__":
    main()
