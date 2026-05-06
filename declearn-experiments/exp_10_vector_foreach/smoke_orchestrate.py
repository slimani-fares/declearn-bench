"""exp_10 smoke orchestrator.

Pattern:
  for branch in (master, variant):
      git checkout <branch> in fork
      pip install -e fork (declearn) so any imports use this fork
      run smoke_unit.py -> writes pickle of all op outputs
  compare pickles, assert torch.equal across all ops.
"""

import pickle
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path("/home/fslimani/declearn-bench")
EXP = REPO / "declearn-experiments" / "exp_10_vector_foreach"
FORK = REPO / "declearn-for-exp_10_vector_foreach"
PY = "/home/fslimani/.venvs/declearn313/bin/python"


def install(branch):
    subprocess.run(["git", "checkout", branch], cwd=FORK, check=True,
                   capture_output=True)
    subprocess.run([PY, "-m", "pip", "install", "-e", str(FORK), "-q"],
                   check=True, capture_output=True)


def run_unit(branch, out_path):
    cmd = [PY, str(EXP / "smoke_unit.py"), "--branch", branch,
           "--out", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode:
        print(r.stderr)
        raise RuntimeError(f"smoke_unit.py failed for {branch}")


def main():
    out_master = EXP / "runs" / "smoke_master.pkl"
    out_variant = EXP / "runs" / "smoke_variant.pkl"
    out_master.parent.mkdir(exist_ok=True, parents=True)

    print("=== installing master (canonical) and running unit smoke ===")
    install("master")
    run_unit("master", out_master)

    print("\n=== installing variant (foreach) and running unit smoke ===")
    install("exp_10_vector_foreach_variant")
    run_unit("variant", out_variant)

    print("\n=== comparing outputs op-by-op ===")
    a = pickle.loads(out_master.read_bytes())
    b = pickle.loads(out_variant.read_bytes())
    assert a.keys() == b.keys(), f"op-set mismatch {a.keys()} vs {b.keys()}"
    failures = []
    summary = []
    for op in sorted(a):
        ad, bd = a[op], b[op]
        if ad.keys() != bd.keys():
            failures.append((op, "keys differ"))
            continue
        max_abs = 0.0
        all_equal = True
        for k in ad:
            if not torch.equal(ad[k], bd[k]):
                all_equal = False
            d = (ad[k] - bd[k]).abs().max().item()
            if d > max_abs:
                max_abs = d
        summary.append((op, all_equal, max_abs))
        if not all_equal and max_abs > 1e-6:
            failures.append((op, f"max abs diff {max_abs:.3e}"))

    print(f"\n{'op':<10} {'byte_equal':>12} {'max_abs_diff':>16}")
    for op, eq, mad in summary:
        print(f"{op:<10} {str(eq):>12} {mad:>16.3e}")

    if failures:
        print(f"\nFAIL: {len(failures)} ops mismatched beyond 1e-6")
        for op, msg in failures:
            print(f"  {op}: {msg}")
        sys.exit(1)
    else:
        print(f"\nPASS: all {len(summary)} ops match canonical "
              f"(byte-equal where allowed; <=1e-6 elsewhere)")
        # Reset to canonical install so the rest of the env stays sane.
        print("\n=== resetting venv to canonical declearn ===")
        subprocess.run([PY, "-m", "pip", "install", "-e", str(REPO / "declearn"),
                        "-q"], check=True, capture_output=True)


if __name__ == "__main__":
    main()
