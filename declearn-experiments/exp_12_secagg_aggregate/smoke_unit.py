"""exp_12 unit smoke: aggregate_encrypted byte-equality master vs variant.

For each branch, build a MaskedAggregate, call aggregate_encrypted on
synthetic random uint64 lists at default max_int=2^64 and a non-default
max_int=2^32, and pickle the outputs. Then compare elementwise.
"""

import pickle
import random
import shutil
import subprocess
import sys
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


def run_branch(branch_label, out_path):
    script = """
import pickle, sys, random
# Reset declearn caches.
for k in list(sys.modules):
    if k.startswith('declearn'):
        del sys.modules[k]
from declearn.secagg.masking._aggregate import MaskedAggregate
# EncryptedSpecs is a `List[Tuple[...]]` type alias; an empty list is valid.
from declearn.utils import Aggregate

class _DummyAgg(Aggregate, register=False):
    pass

random.seed(123)
L = 50_000
# default max_int = 2**64 → uint64 natural wrap path
vals_64a = [random.getrandbits(64) for _ in range(L)]
vals_64b = [random.getrandbits(64) for _ in range(L)]
# smaller max_int = 2**32 → explicit modulo path
vals_32a = [random.getrandbits(32) for _ in range(L)]
vals_32b = [random.getrandbits(32) for _ in range(L)]

# Build a MaskedAggregate (it just needs the right max_int; encrypted/specs
# can be empty stubs because aggregate_encrypted only operates on its args).
agg64 = MaskedAggregate(encrypted=[], enc_specs=[], cleartext=None,
                        agg_cls=_DummyAgg, max_int=2**64, n_aggrg=1)
agg32 = MaskedAggregate(encrypted=[], enc_specs=[], cleartext=None,
                        agg_cls=_DummyAgg, max_int=2**32, n_aggrg=1)

import time
t0 = time.perf_counter()
out_64 = agg64.aggregate_encrypted(vals_64a, vals_64b)
t_64 = time.perf_counter() - t0
t0 = time.perf_counter()
out_32 = agg32.aggregate_encrypted(vals_32a, vals_32b)
t_32 = time.perf_counter() - t0

OUT = sys.argv[1]
import pathlib
pathlib.Path(OUT).write_bytes(pickle.dumps({
    'inputs_64': (vals_64a, vals_64b),
    'inputs_32': (vals_32a, vals_32b),
    'out_64': out_64,
    'out_32': out_32,
    't_64_s': t_64,
    't_32_s': t_32,
}))
print(f'[{sys.argv[2]}] L={L}  max_int=2**64: {t_64*1000:.1f} ms  max_int=2**32: {t_32*1000:.1f} ms')
"""
    cmd = [PY, "-c", script, str(out_path), branch_label]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode:
        print("STDERR:", r.stderr)
        raise RuntimeError(f"branch {branch_label} failed")


def main():
    out_master = EXP / "runs" / "smoke_unit_master.pkl"
    out_variant = EXP / "runs" / "smoke_unit_variant.pkl"
    out_master.parent.mkdir(exist_ok=True)

    print("=== exp_12 unit smoke ===\n")
    print("--- master ---")
    install("master")
    run_branch("master", out_master)

    print("\n--- variant ---")
    install("exp_12_secagg_aggregate_variant")
    run_branch("variant", out_variant)

    print("\n=== resetting venv ===")
    subprocess.run([PY, "-m", "pip", "install", "-e", str(REPO / "declearn"),
                    "-q"], check=True, capture_output=True)

    a = pickle.loads(out_master.read_bytes())
    b = pickle.loads(out_variant.read_bytes())

    # Inputs were re-seeded to be identical; verify.
    assert a['inputs_64'] == b['inputs_64'], "input mismatch"
    assert a['inputs_32'] == b['inputs_32'], "input mismatch"

    # Element-wise equality for outputs.
    eq_64 = a['out_64'] == b['out_64']
    eq_32 = a['out_32'] == b['out_32']
    print(f"\nmax_int=2^64: master timing {a['t_64_s']*1000:.1f} ms, "
          f"variant {b['t_64_s']*1000:.1f} ms ({a['t_64_s']/b['t_64_s']:.2f}× speedup)")
    print(f"             output equality: {eq_64}")
    print(f"max_int=2^32: master timing {a['t_32_s']*1000:.1f} ms, "
          f"variant {b['t_32_s']*1000:.1f} ms ({a['t_32_s']/b['t_32_s']:.2f}× speedup)")
    print(f"             output equality: {eq_32}")

    if not (eq_64 and eq_32):
        print("FAIL: outputs differ from canonical")
        # Show first 5 mismatches.
        if not eq_64:
            for i, (x, y) in enumerate(zip(a['out_64'], b['out_64'])):
                if x != y:
                    print(f"  64-bit mismatch at idx {i}: master={x} variant={y}")
                    if i > 3:
                        break
        sys.exit(1)
    print("\nPASS: outputs byte-equal at both max_int values")


if __name__ == "__main__":
    main()
