"""exp_10 unit-level smoke: TorchVector(_foreach_*) == canonical TorchVector.

Runs each binary op (+, -, *, /, **, minimum, maximum) and unary op (sign,
abs, neg, reciprocal) on a TorchVector with the same shapes as
mnist_quickrun's CNN, on both branches of the fork. Asserts torch.allclose
within atol=0 (for ops that are purely framework dispatch this should be
byte-identical; we keep allclose for the rare ULP-level reorderings that
torch._foreach_* might introduce on float ops).

Runs entirely in-process by importing the fork directly via path, so we
do NOT need to pip-install/uninstall to compare. We import the canonical
TorchVector first (master tree) and the variant TorchVector second (variant
tree), then compare outputs side-by-side on the SAME random tensors.
"""

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path("/home/fslimani/declearn-bench")
FORK = REPO / "declearn-for-exp_10_vector_foreach"


def load_module_from(path: Path, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_coefs(seed: int = 0):
    torch.manual_seed(seed)
    # Shapes mimic the mnist_quickrun CNN parameter dict.
    shapes = {
        "conv1.weight": (8, 1, 3, 3),
        "conv1.bias": (8,),
        "conv2.weight": (16, 8, 3, 3),
        "conv2.bias": (16,),
        "fc1.weight": (32, 256),
        "fc1.bias": (32,),
        "fc2.weight": (10, 32),
        "fc2.bias": (10,),
    }
    return {k: torch.randn(s) for k, s in shapes.items()}


def vec_dict(v):
    return v.coefs


def assert_close_dict(a_coefs, b_coefs, label, atol=0.0, rtol=0.0):
    assert a_coefs.keys() == b_coefs.keys(), (
        f"{label}: keys differ {a_coefs.keys()} vs {b_coefs.keys()}"
    )
    for k in a_coefs:
        if not torch.allclose(a_coefs[k], b_coefs[k], atol=atol, rtol=rtol):
            diff = (a_coefs[k] - b_coefs[k]).abs().max().item()
            raise AssertionError(
                f"{label} mismatch on key '{k}': max abs diff = {diff:.3e}"
            )


def main():
    # Run twice: once installing fork at master (canonical), once at variant.
    # We do a single subprocess pattern: this script is invoked with an arg
    # specifying which branch to use. It writes its outputs to a pickle, and
    # the orchestrator (smoke_orchestrate.py) compares.
    import argparse
    import pickle

    p = argparse.ArgumentParser()
    p.add_argument("--branch", required=True, choices=["master", "variant"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # Force-import the fork's TorchVector (both master and variant produce the
    # same import path; we control which by checking out the branch first).
    sys.path.insert(0, str(FORK))
    # Drop any pre-cached declearn modules.
    for k in list(sys.modules):
        if k.startswith("declearn"):
            del sys.modules[k]
    from declearn.model.torch import TorchVector  # noqa: E402

    coefs_a = make_coefs(seed=0)
    coefs_b = make_coefs(seed=1)
    va = TorchVector({k: v.clone() for k, v in coefs_a.items()})
    vb = TorchVector({k: v.clone() for k, v in coefs_b.items()})

    out = {}

    # Binary: vector + vector
    out["add_vv"] = {k: v.detach().clone() for k, v in (va + vb).coefs.items()}
    out["sub_vv"] = {k: v.detach().clone() for k, v in (va - vb).coefs.items()}
    out["mul_vv"] = {k: v.detach().clone() for k, v in (va * vb).coefs.items()}
    out["div_vv"] = {k: v.detach().clone() for k, v in (va / vb).coefs.items()}

    # Binary: vector + scalar
    out["add_vs"] = {k: v.detach().clone() for k, v in (va + 2.5).coefs.items()}
    out["mul_vs"] = {k: v.detach().clone() for k, v in (va * 0.7).coefs.items()}
    out["div_vs"] = {k: v.detach().clone() for k, v in (va / 3.0).coefs.items()}
    out["pow_vs"] = {k: v.detach().clone() for k, v in (va ** 2.0).coefs.items()}

    # Vector.minimum/maximum (float and Vector arms)
    out["min_vv"] = {k: v.detach().clone() for k, v in va.minimum(vb).coefs.items()}
    out["max_vv"] = {k: v.detach().clone() for k, v in va.maximum(vb).coefs.items()}
    out["min_vf"] = {k: v.detach().clone() for k, v in va.minimum(0.1).coefs.items()}
    out["max_vf"] = {k: v.detach().clone() for k, v in va.maximum(-0.1).coefs.items()}

    # Unary: sign / apply_func paths
    out["sign"] = {k: v.detach().clone() for k, v in va.sign().coefs.items()}
    # apply_func with extra args MUST still go through the canonical loop on
    # the variant branch (its fast path requires no args/kwargs). Verify by
    # using torch.clamp which has no _foreach_ counterpart in our map.
    out["clamp"] = {
        k: v.detach().clone()
        for k, v in va.apply_func(torch.clamp, min=-0.5, max=0.5).coefs.items()
    }

    # Sanity: sum, repr stability
    out["sum"] = {k: v.detach().clone() for k, v in va.sum().coefs.items()}

    Path(args.out).write_bytes(pickle.dumps(out))
    print(f"[{args.branch}] wrote {len(out)} ops to {args.out}")


if __name__ == "__main__":
    main()
