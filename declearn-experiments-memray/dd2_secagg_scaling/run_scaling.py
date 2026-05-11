"""Run memray master + variant on the DD2 SecAgg-masking patch at N=50, N=100.

Sequential execution to avoid resource contention. Emits one stdout line
per run completion so progress can be monitored externally.
"""

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/fslimani/declearn-bench")
ROOT = REPO / "declearn-experiments-memray" / "dd2_secagg_scaling"
FORK = REPO / "declearn-for-exp_12_secagg_aggregate"
PY = "/home/fslimani/.venvs/declearn313/bin/python"
RUN_MEMRAY = REPO / "declearn-experiments-memray" / "_setup" / "run_memray.py"

N_VALUES = [50, 100]
ARMS = [
    ("master", "master"),
    ("variant", "dd2_json_binary_variant"),
]


def install(branch):
    subprocess.run(["git", "checkout", branch], cwd=FORK, check=True,
                   capture_output=True)
    subprocess.run([PY, "-m", "pip", "install", "-e", str(FORK), "-q"],
                   check=True, capture_output=True)


def run_one(n, arm_label, branch):
    cfg = ROOT / "configs" / f"config_secagg_n{n}.toml"
    out = ROOT / "runs"
    target = (
        f"import asyncio; from declearn.quickrun._run import quickrun; "
        f"asyncio.run(quickrun({str(cfg)!r}, secagg_variant='masking'))"
    )
    # Wipe stale checkpoints.
    for p in ROOT.glob("result_*"):
        shutil.rmtree(p, ignore_errors=True)
    cmd = [
        PY, str(RUN_MEMRAY),
        "--out", str(out),
        "--tag", f"n{n}_{arm_label}",
        "--target", target,
    ]
    print(f"[start] N={n} arm={arm_label} branch={branch}", flush=True)
    install(branch)
    start = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dur = time.perf_counter() - start
    print(f"[done ] N={n} arm={arm_label} dur={dur:.1f}s rc={r.returncode}", flush=True)
    if r.returncode:
        print(f"  STDOUT tail: {r.stdout.strip()[-400:]}", flush=True)
        print(f"  STDERR tail: {r.stderr.strip()[-400:]}", flush=True)
        return None
    # Locate the most recent run dir for this arm and pull stats.
    arm_dir = out / f"n{n}_{arm_label}"
    latest = sorted(arm_dir.iterdir())[-1]
    summary_path = latest / "summary.txt"
    txt = summary_path.read_text()
    out_stats = {}
    m = re.search(r"Total allocations:\s*\n\s*(\d+)", txt)
    if m: out_stats["n_allocs"] = int(m.group(1))
    m = re.search(r"Total memory allocated:\s*\n\s*([\d.]+)([KMG]?)B", txt)
    if m:
        v, u = float(m.group(1)), m.group(2)
        out_stats["total_mb"] = v * {"": 1e-6, "K": 1e-3, "M": 1, "G": 1000}[u]
    m = re.search(r"Peak memory usage:\s*\n\s*([\d.]+)([KMG]?)B", txt)
    if m:
        v, u = float(m.group(1)), m.group(2)
        out_stats["peak_mb"] = v * {"": 1e-6, "K": 1e-3, "M": 1, "G": 1000}[u]
    out_stats["summary"] = str(summary_path)
    out_stats["flame"] = str(latest / "flamegraph.html")
    out_stats["wall_s"] = round(dur, 2)
    print(f"  peak={out_stats.get('peak_mb', 0):.1f}MB  "
          f"total={out_stats.get('total_mb', 0):.0f}MB  "
          f"summary={summary_path}", flush=True)
    return out_stats


def main():
    print(f"=== DD2 scaling: N={N_VALUES} × {len(ARMS)} arms ===", flush=True)
    results = {}
    for n in N_VALUES:
        results[n] = {}
        for arm_label, branch in ARMS:
            try:
                stats = run_one(n, arm_label, branch)
                if stats:
                    results[n][arm_label] = stats
            except Exception as exc:
                print(f"[fail ] N={n} arm={arm_label}: {exc}", flush=True)

    # Reset venv.
    print("\n=== resetting venv to canonical declearn ===", flush=True)
    subprocess.run([PY, "-m", "pip", "install", "-e", str(REPO / "declearn"),
                    "-q"], check=True, capture_output=True)

    # Print comparison
    print("\n=== summary ===", flush=True)
    print(f"{'N':>4} {'arm':<8} {'peak_MB':>10} {'total_MB':>12} {'n_allocs':>14} {'wall_s':>8}",
          flush=True)
    for n in N_VALUES:
        for arm_label, _ in ARMS:
            s = results.get(n, {}).get(arm_label)
            if s is None:
                print(f"{n:>4} {arm_label:<8}  FAILED", flush=True)
                continue
            print(f"{n:>4} {arm_label:<8} {s.get('peak_mb', 0):>10.1f} "
                  f"{s.get('total_mb', 0):>12.0f} {s.get('n_allocs', 0):>14,} "
                  f"{s.get('wall_s', 0):>8.1f}", flush=True)

    # JSON dump for the recap step
    import json
    json_out = ROOT / "scaling_results.json"
    json_out.write_text(json.dumps({str(n): r for n, r in results.items()},
                                    indent=2))
    print(f"\nresults json: {json_out}", flush=True)


if __name__ == "__main__":
    main()
