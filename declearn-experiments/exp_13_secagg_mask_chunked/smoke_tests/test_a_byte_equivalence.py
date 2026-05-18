"""Smoke test A — byte-equivalence of mask streams.

Compares the canonical _generate_masks_numpy (in ~/declearn-bench/declearn/)
against the chunked variant (in ~/declearn-bench/declearn-for-exp_13_secagg_mask_chunked/).

For default bitsize=64, dtype=uint64, max_int = 2**64, numpy.random.Generator.integers
runs without rejection sampling (full-range uint64), so chunked draws should
emit byte-identical mask streams given the same seeds.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


def _load_encrypter(path: Path, alias: str):
    sys.path.insert(0, str(path))
    if "declearn" in sys.modules:
        # Drop any cached declearn modules so re-imports come from `path`.
        for mod in list(sys.modules):
            if mod == "declearn" or mod.startswith("declearn."):
                del sys.modules[mod]
    mod = importlib.import_module("declearn.secagg.masking._encrypt")
    sys.modules[f"_alias_{alias}"] = mod
    sys.path.pop(0)
    return mod.MaskingEncrypter


def main() -> int:
    # master arm is exp_12 master (declearn 2.8.0 + unified runner + batched
    # encrypt-side, but canonical _generate_masks_numpy). variant arm is
    # exp_13 chunked _generate_masks_numpy on top of that same base.
    base_path = Path.home() / "declearn-bench" / "declearn-for-exp_12_secagg_aggregate"
    var_path = Path.home() / "declearn-bench" / "declearn-for-exp_13_secagg_mask_chunked"

    Master = _load_encrypter(base_path, "master")
    Variant = _load_encrypter(var_path, "variant")

    # Seeds must be identical for byte-equivalence claim.
    pos_seeds = [11, 22, 33]
    neg_seeds = [44, 55]
    n_values = 1_000_000  # large enough that chunking kicks in (CHUNK=65536)

    master = Master(pos_masks_seeds=pos_seeds, neg_masks_seeds=neg_seeds, bitsize=64)
    variant = Variant(pos_masks_seeds=pos_seeds, neg_masks_seeds=neg_seeds, bitsize=64)

    mask_master = master._generate_masks_numpy(n_values)
    mask_variant = variant._generate_masks_numpy(n_values)

    assert mask_master.dtype == mask_variant.dtype == np.uint64, (
        f"dtype mismatch: master={mask_master.dtype}, variant={mask_variant.dtype}"
    )
    assert mask_master.shape == mask_variant.shape == (n_values,), (
        f"shape mismatch: master={mask_master.shape}, variant={mask_variant.shape}"
    )
    equal = np.array_equal(mask_master, mask_variant)
    print(f"n_values={n_values}, dtype={mask_master.dtype}, byte-identical={equal}")
    if not equal:
        diff_idx = np.where(mask_master != mask_variant)[0]
        print(f"  diff count: {len(diff_idx)}")
        print(f"  first 5 diff indices: {diff_idx[:5].tolist()}")
        print(f"  first 5 master: {mask_master[diff_idx[:5]].tolist()}")
        print(f"  first 5 variant: {mask_variant[diff_idx[:5]].tolist()}")
        return 1

    # Small-n smoke: when n_values < CHUNK, behaviour must still match.
    mask_master_s = master._generate_masks_numpy(10)
    mask_variant_s = variant._generate_masks_numpy(10)
    assert np.array_equal(mask_master_s, mask_variant_s), "small-n byte mismatch"
    print(f"small-n (n=10) byte-identical: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
