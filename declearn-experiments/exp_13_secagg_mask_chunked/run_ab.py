"""exp_13 memray A/B: canonical declearn vs chunked _generate_masks_numpy.

Sweeps N in {5, 20, 50, 100} x {master, variant_h1_chunked}. One memray
run each, 1 round, mnist quickrun + SecAgg masking.
"""

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/fslimani/declearn-bench")
ROOT = REPO / "declearn-experiments" / "exp_13_secagg_mask_chunked"
PY = "/home/fslimani/.venvs/declearn313/bin/python"
RUN_MEMRAY = REPO / "declearn-experiments-memray" / "_setup" / "run_memray.py"

# master arm = exp_12's master branch (canonical declearn 2.8.0 + unified
# runner + batched encrypt-side from exp_06 fork). This is the meaningful
# baseline against which a NEW intervention should be measured: "current
# declearn with everything Fares has shipped so far".
MASTER_FORK = REPO / "declearn-for-exp_12_secagg_aggregate"
# variant arm = exp_12 master + chunked _generate_masks_numpy patch
VARIANT_FORK = REPO / "declearn-for-exp_13_secagg_mask_chunked"

N_VALUES = [5, 20, 50, 100]
ARMS = [
    ("master", MASTER_FORK, "master"),
    ("variant", VARIANT_FORK, "exp_13_variant_h1_chunked"),
]


def install(path, branch=None):
    if branch is not None:
        subprocess.run(
            ["git", "checkout", branch], cwd=path, check=True,
            capture_output=True,
        )
    subprocess.run(
        [PY, "-m", "pip", "install", "-e", str(path), "-q"],
        check=True, capture_output=True,
    )


def parse_summary(summary_path):
    txt = summary_path.read_text()
    out = {}
    m = re.search(r"Total allocations:\s*\n\s*([\d,]+)", txt)
    if m:
        out["n_allocs"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Total memory allocated:\s*\n\s*([\d.]+)([KMG]?)B", txt)
    if m:
        v, u = float(m.group(1)), m.group(2)
        out["total_mb"] = v * {"": 1e-6, "K": 1e-3, "M": 1, "G": 1000}[u]
    m = re.search(r"Peak memory usage:\s*\n\s*([\d.]+)([KMG]?)B", txt)
    if m:
        v, u = float(m.group(1)), m.group(2)
        out["peak_mb"] = v * {"": 1e-6, "K": 1e-3, "M": 1, "G": 1000}[u]
    return out


def run_one(n, arm_label, fork_path, branch):
    cfg = ROOT / "configs" / f"config_secagg_n{n}.toml"
    out = ROOT / "runs"
    target = (
        f"import asyncio; from declearn.quickrun._run import quickrun; "
        f"asyncio.run(quickrun({str(cfg)!r}, secagg_variant='masking'))"
    )
    # Wipe stale checkpoint dirs lying around in the repo (quickrun writes
    # result_* near cwd when no explicit dir is set).
    for p in REPO.glob("result_*"):
        shutil.rmtree(p, ignore_errors=True)
    for p in ROOT.glob("result_*"):
        shutil.rmtree(p, ignore_errors=True)
    cmd = [
        PY, str(RUN_MEMRAY),
        "--out", str(out),
        "--tag", f"n{n}_{arm_label}",
        "--target", target,
    ]
    print(f"[start] N={n} arm={arm_label} (fork={fork_path.name}, branch={branch})",
          flush=True)
    install(fork_path, branch)
    start = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    dur = time.perf_counter() - start
    print(f"[done ] N={n} arm={arm_label} dur={dur:.1f}s rc={r.returncode}",
          flush=True)
    if r.returncode:
        print(f"  STDOUT tail: {r.stdout.strip()[-500:]}", flush=True)
        print(f"  STDERR tail: {r.stderr.strip()[-500:]}", flush=True)
        return None
    arm_dir = out / f"n{n}_{arm_label}"
    latest = sorted(arm_dir.iterdir())[-1]
    summary_path = latest / "summary.txt"
    stats = parse_summary(summary_path)
    stats["summary"] = str(summary_path)
    stats["flame"] = str(latest / "flamegraph.html")
    stats["wall_s"] = round(dur, 2)
    stats["arm_dir"] = str(latest)
    print(f"  peak={stats.get('peak_mb', 0):.1f}MB "
          f"total={stats.get('total_mb', 0):.0f}MB "
          f"n_allocs={stats.get('n_allocs', 0):,}",
          flush=True)
    return stats


def main():
    print(f"=== exp_13 A/B: N={N_VALUES} x {len(ARMS)} arms ===", flush=True)
    results = {}
    for n in N_VALUES:
        results[n] = {}
        for arm_label, fork_path, branch in ARMS:
            try:
                stats = run_one(n, arm_label, fork_path, branch)
                if stats:
                    results[n][arm_label] = stats
            except Exception as exc:
                print(f"[fail ] N={n} arm={arm_label}: {exc}", flush=True)

    # Reset venv to canonical declearn (post-experiment cleanup).
    print("\n=== resetting venv to canonical declearn ===", flush=True)
    install(REPO / "declearn")

    print("\n=== summary ===", flush=True)
    print(f"{'N':>4} {'arm':<8} {'peak_MB':>10} {'total_MB':>12} "
          f"{'n_allocs':>14} {'wall_s':>8}",
          flush=True)
    for n in N_VALUES:
        for arm_label, _, _ in ARMS:
            s = results.get(n, {}).get(arm_label)
            if s is None:
                print(f"{n:>4} {arm_label:<8}  FAILED", flush=True)
                continue
            print(f"{n:>4} {arm_label:<8} {s.get('peak_mb', 0):>10.1f} "
                  f"{s.get('total_mb', 0):>12.0f} "
                  f"{s.get('n_allocs', 0):>14,} "
                  f"{s.get('wall_s', 0):>8.1f}", flush=True)

    import json
    json_out = ROOT / "ab_results.json"
    json_out.write_text(json.dumps({str(n): r for n, r in results.items()},
                                    indent=2))
    print(f"\nresults: {json_out}", flush=True)


if __name__ == "__main__":
    main()
