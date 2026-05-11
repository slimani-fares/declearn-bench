"""exp_M06 follow-up: extend msgpack A/B to N in {20, 50, 100}.

Same workload, same config shape (rounds=5, n_steps=5, vanilla averaging,
no SecAgg). Only N (and register timeout/min_clients) varies.

Sequential per-cell execution. One pip install per cell.
"""

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path("/home/fslimani/declearn-bench")
EXP = REPO / "declearn-experiments-memray" / "exp_M06_msgpack_vanilla"
UPSTREAM = REPO / "_scratch" / "declearn_upstream"
PY = "/home/fslimani/.venvs/declearn313/bin/python"
RUN_MEMRAY = REPO / "declearn-experiments-memray" / "_setup" / "run_memray.py"

N_VALUES = [20, 50, 100]
REPEATS = [1, 2]
CELLS = [
    ("baseline", "9668576"),
    ("msgpack",  "d402885b"),
]


def install(commit):
    subprocess.run(["git", "checkout", commit], cwd=UPSTREAM, check=True,
                   capture_output=True)
    subprocess.run([PY, "-m", "pip", "install", "-e", str(UPSTREAM), "-q"],
                   check=True, capture_output=True)


def run_one(cell, commit, n, repeat):
    cfg = EXP / f"config_n{n}.toml"
    target = (
        f"import asyncio; from declearn.quickrun._run import quickrun; "
        f"asyncio.run(quickrun({str(cfg)!r}))"
    )
    for root in (EXP, REPO):
        for p in root.glob("result_*"):
            shutil.rmtree(p, ignore_errors=True)
    out = EXP / "runs" / cell
    out.mkdir(parents=True, exist_ok=True)
    tag = f"n{n}_rep{repeat}"
    cmd = [
        PY, str(RUN_MEMRAY),
        "--out", str(out),
        "--tag", tag,
        "--target", target,
    ]
    print(f"[start] cell={cell} N={n} rep={repeat} commit={commit}", flush=True)
    start = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dur = time.perf_counter() - start
    print(f"[done ] cell={cell} N={n} rep={repeat} dur={dur:.1f}s rc={r.returncode}",
          flush=True)
    if r.returncode:
        print(f"  STDOUT tail: {r.stdout.strip()[-400:]}", flush=True)
        print(f"  STDERR tail: {r.stderr.strip()[-400:]}", flush=True)
        return None
    arm_dir = out / tag
    latest = sorted(arm_dir.iterdir())[-1]
    summary_path = latest / "summary.txt"
    txt = summary_path.read_text()
    s = {"wall_s": round(dur, 2),
         "summary": str(summary_path),
         "flame": str(latest / "flamegraph.html")}
    m = re.search(r"Total allocations:\s*\n\s*(\d+)", txt)
    if m: s["n_allocs"] = int(m.group(1))
    m = re.search(r"Total memory allocated:\s*\n\s*([\d.]+)([KMG]?)B", txt)
    if m: s["total_mb"] = float(m.group(1)) * {"": 1e-6, "K": 1e-3, "M": 1, "G": 1000}[m.group(2)]
    m = re.search(r"Peak memory usage:\s*\n\s*([\d.]+)([KMG]?)B", txt)
    if m: s["peak_mb"] = float(m.group(1)) * {"": 1e-6, "K": 1e-3, "M": 1, "G": 1000}[m.group(2)]
    s["iterencode_mb"] = 0.0
    s["raw_decode_mb"] = 0.0
    for m in re.finditer(r"-\s*(iterencode|raw_decode):[^\s]+\s*->\s*([\d.]+)([KMG]?)B", txt):
        name = m.group(1)
        v = float(m.group(2)) * {"": 1e-6, "K": 1e-3, "M": 1, "G": 1000}[m.group(3)]
        s[f"{name}_mb"] = v
    print(f"  peak={s.get('peak_mb', 0):.1f}MB total={s.get('total_mb', 0):.0f}MB "
          f"iterencode={s.get('iterencode_mb', 0):.1f}MB "
          f"raw_decode={s.get('raw_decode_mb', 0):.1f}MB",
          flush=True)
    return s


def main():
    print(f"=== M06 msgpack A/B bigN: {len(CELLS)} cells x {len(N_VALUES)} N x "
          f"{len(REPEATS)} repeats ===", flush=True)
    results = {cell: {} for cell, _ in CELLS}
    for cell, commit in CELLS:
        print(f"\n--- installing {cell} @ {commit} ---", flush=True)
        install(commit)
        for n in N_VALUES:
            results[cell][n] = []
            for r in REPEATS:
                s = run_one(cell, commit, n, r)
                if s is not None:
                    results[cell][n].append(s)

    print("\n=== resetting venv to canonical declearn ===", flush=True)
    subprocess.run([PY, "-m", "pip", "install", "-e", str(REPO / "declearn"),
                    "-q"], check=True, capture_output=True)

    out_json = EXP / "ab_results_bigN.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nresults json: {out_json}", flush=True)

    print(f"\n=== summary bigN (mean of {len(REPEATS)} repeats) ===", flush=True)
    print(f"{'cell':<10} {'N':>3} {'peak_MB':>10} {'total_MB':>10} "
          f"{'iter_MB':>9} {'rawdec_MB':>10} {'wall_s':>8}", flush=True)
    for cell, _ in CELLS:
        for n in N_VALUES:
            rs = results[cell][n]
            if not rs:
                print(f"{cell:<10} {n:>3}  FAILED", flush=True)
                continue
            avg = lambda k: sum(r.get(k, 0) for r in rs) / len(rs)
            print(f"{cell:<10} {n:>3} {avg('peak_mb'):>10.1f} {avg('total_mb'):>10.0f} "
                  f"{avg('iterencode_mb'):>9.1f} {avg('raw_decode_mb'):>10.1f} "
                  f"{avg('wall_s'):>8.1f}", flush=True)


if __name__ == "__main__":
    main()
