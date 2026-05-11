"""Generate examples/mnist_quickrun/data_iid_fair (3-client, fairness-augmented)
from data_iid. Reproduces _ensure_data_fair() from benchmarks/__init__.py.backup.

Idempotent: skips if data_iid_fair already exists.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/fslimani/declearn-bench")
SRC_FOLDER = REPO / "examples" / "mnist_quickrun"
PY = "/home/fslimani/.venvs/declearn313/bin/python"


def regenerate_iid_split(seed=42, n_shards=3):
    """Regen data_iid (3 clients) deterministically."""
    for d in SRC_FOLDER.glob("data_iid*"):
        if d.is_dir() and d.name == "data_iid":
            shutil.rmtree(d, ignore_errors=True)
    subprocess.run(
        ["/home/fslimani/.venvs/declearn313/bin/declearn-split",
         "--folder", str(SRC_FOLDER.relative_to(REPO)),
         "--n_shards", str(n_shards), "--scheme", "iid", "--seed", str(seed)],
        cwd=REPO, check=True, capture_output=True,
    )


def build_data_iid_fair():
    """Binarize labels (digit >= 5) and add s_attr (digit % 2)."""
    src_root = SRC_FOLDER / "data_iid"
    dst_root = SRC_FOLDER / "data_iid_fair"
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir()
    for client_dir in sorted(src_root.iterdir()):
        if not client_dir.is_dir():
            continue
        out = dst_root / client_dir.name
        out.mkdir()
        for split in ("train", "valid"):
            shutil.copy(client_dir / f"{split}_data.npy", out / f"{split}_data.npy")
            tgt = np.load(client_dir / f"{split}_target.npy")
            bin_tgt = (tgt >= 5).astype(np.int64)
            s_attr = (tgt % 2).astype(np.float32).reshape(-1, 1)
            np.save(out / f"{split}_target.npy", bin_tgt)
            np.save(out / f"{split}_s_attr.npy", s_attr)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    n_shards = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print(f"Regenerating data_iid (n_shards={n_shards}, seed={seed})...")
    regenerate_iid_split(seed=seed, n_shards=n_shards)
    print("Building data_iid_fair (binary target + s_attr)...")
    build_data_iid_fair()
    # Sanity probe
    p = SRC_FOLDER / "data_iid_fair" / "client_0" / "train_target.npy"
    t = np.load(p)
    s = np.load(p.parent / "train_s_attr.npy")
    print(f"client_0 train: target shape={t.shape} unique={np.unique(t).tolist()}")
    print(f"client_0 train: s_attr shape={s.shape} unique={np.unique(s).tolist()}")
    print(f"Output: {SRC_FOLDER / 'data_iid_fair'}")


if __name__ == "__main__":
    main()
