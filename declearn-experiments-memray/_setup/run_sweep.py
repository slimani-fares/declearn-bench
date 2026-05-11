"""Run all 5 memray sweep experiments sequentially.

Each experiment:
  1. Install the right declearn fork (canonical or specific patched fork).
  2. Invoke run_memray.py with the experiment's config + secagg flag.
  3. Collect peak / total / retained from the generated summary.txt.

Outputs a sweep_summary.json with the per-experiment numbers and prints
a comparison table.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/fslimani/declearn-bench")
ROOT = REPO / "declearn-experiments-memray"
PY = "/home/fslimani/.venvs/declearn313/bin/python"

EXPERIMENTS = [
    {
        "tag": "M01_vanilla",
        "config": ROOT / "exp_M01_vanilla" / "config.toml",
        "out": ROOT / "exp_M01_vanilla" / "runs",
        "fork": REPO / "declearn",  # canonical
        "secagg": None,
    },
    {
        "tag": "M02_dp",
        "config": ROOT / "exp_M02_dp" / "config.toml",
        "out": ROOT / "exp_M02_dp" / "runs",
        "fork": REPO / "declearn",
        "secagg": None,
    },
    {
        "tag": "M03_secagg_n20",
        "config": ROOT / "exp_M03_secagg_n20" / "config.toml",
        "out": ROOT / "exp_M03_secagg_n20" / "runs",
        "fork": REPO / "declearn-for-secagg-batched",  # has unified runner + batched encrypt
        "secagg": "masking",
    },
    {
        "tag": "M04_fairgrad",
        "config": ROOT / "exp_M04_fairgrad" / "config.toml",
        "out": ROOT / "exp_M04_fairgrad" / "runs",
        "fork": REPO / "declearn-for-exp_07_fairness",  # unified runner + fairness patch
        "secagg": None,
    },
    {
        "tag": "M05_vector_big",
        "config": ROOT / "exp_M05_vector_big" / "config.toml",
        "out": ROOT / "exp_M05_vector_big" / "runs",
        # Use the variant branch of the exp_10 fork (foreach patch applied)
        "fork": REPO / "declearn-for-exp_10_vector_foreach",
        "fork_branch": "exp_10_vector_foreach_variant",
        "secagg": None,
    },
]


def install(fork_path, branch=None):
    if branch is not None:
        # Local fork with its own git history.
        subprocess.run(["git", "checkout", branch], cwd=fork_path,
                       check=True, capture_output=True)
    subprocess.run([PY, "-m", "pip", "install", "-e", str(fork_path), "-q"],
                   check=True, capture_output=True)


def run_one(exp):
    tag = exp["tag"]
    cfg = exp["config"]
    out = exp["out"]
    secagg = exp["secagg"]
    out.mkdir(parents=True, exist_ok=True)

    secagg_arg = (f", secagg_variant={secagg!r}" if secagg else "")
    target = (
        f"import asyncio; from declearn.quickrun._run import quickrun; "
        f"asyncio.run(quickrun({str(cfg)!r}{secagg_arg}))"
    )

    # Wipe stale checkpoints in the experiment folder.
    exp_dir = cfg.parent
    for p in exp_dir.glob("result_*"):
        import shutil
        shutil.rmtree(p, ignore_errors=True)

    cmd = [
        PY, str(ROOT / "_setup" / "run_memray.py"),
        "--out", str(out),
        "--tag", tag,
        "--target", target,
    ]
    print(f"\n=== {tag} ===")
    print(f"  fork: {exp['fork']}")
    if exp.get("fork_branch"):
        print(f"  branch: {exp['fork_branch']}")
    print(f"  secagg: {secagg}")
    install(exp["fork"], exp.get("fork_branch"))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode:
        print(f"  STDERR: {r.stderr.strip()}")
        return {"tag": tag, "ok": False, "err": r.stderr.strip()}
    # Find the generated summary.txt
    latest_dir = sorted((out / tag).iterdir())[-1]
    summary_path = latest_dir / "summary.txt"
    metadata_path = latest_dir / "metadata.json"
    return {
        "tag": tag,
        "ok": True,
        "summary": summary_path,
        "metadata": metadata_path,
        "flame": latest_dir / "flamegraph.html",
    }


def parse_summary(summary_path):
    """Pull peak / total / total-allocations from memray stats output."""
    txt = summary_path.read_text()
    out = {}
    m = re.search(r"Total allocations:\s*\n\s*(\d+)", txt)
    if m:
        out["total_allocations"] = int(m.group(1))
    m = re.search(r"Total memory allocated:\s*\n\s*([\d.]+)([KMG]?)B", txt)
    if m:
        val = float(m.group(1))
        u = m.group(2)
        out["total_memory_mb"] = val * {"": 1e-6, "K": 1e-3, "M": 1, "G": 1000}[u]
    m = re.search(r"Peak memory usage:\s*\n\s*([\d.]+)([KMG]?)B", txt)
    if m:
        val = float(m.group(1))
        u = m.group(2)
        out["peak_memory_mb"] = val * {"": 1e-6, "K": 1e-3, "M": 1, "G": 1000}[u]
    return out


def main():
    print(f"=== memray sweep: {len(EXPERIMENTS)} experiments ===\n")
    results = []
    for exp in EXPERIMENTS:
        r = run_one(exp)
        if r["ok"]:
            stats = parse_summary(r["summary"])
            r["stats"] = stats
        results.append(r)

    # Reset to canonical declearn.
    print("\n=== resetting venv to canonical declearn ===")
    subprocess.run([PY, "-m", "pip", "install", "-e", str(REPO / "declearn"),
                    "-q"], check=True, capture_output=True)

    # Save and print summary.
    out_path = ROOT / "sweep_results.json"
    serialized = []
    for r in results:
        rs = {k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()}
        serialized.append(rs)
    out_path.write_text(json.dumps(serialized, indent=2))
    print(f"\n=== sweep results -> {out_path} ===\n")

    print(f"{'tag':<22} {'peak_MB':>10} {'total_alloc':>14} {'n_allocs':>14}")
    for r in results:
        if not r.get("ok"):
            print(f"{r['tag']:<22} FAILED — {r.get('err', '')[:60]}")
            continue
        s = r.get("stats", {})
        peak = s.get("peak_memory_mb", float("nan"))
        total = s.get("total_memory_mb", float("nan"))
        n = s.get("total_allocations", 0)
        print(f"{r['tag']:<22} {peak:>10.1f} {total:>13.1f}MB {n:>14,}")

    print("\nflamegraphs:")
    for r in results:
        if r.get("ok"):
            print(f"  {r['tag']:<22} {r['flame']}")


if __name__ == "__main__":
    main()
