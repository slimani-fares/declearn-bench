"""exp_11 unit-level smoke: directly compare FairnessMetricsComputer output
between master and variant. Cleaner than log parsing and lets us assert
byte-equivalence (modulo float-reorder noise) on the actual return values.

For each branch:
  1. Build a FairnessInMemoryDataset from the data_iid_fair client_0 split.
  2. Build a TorchModel + a deterministic loss/metric setup.
  3. Initialize random weights with a pinned seed.
  4. Call computer.compute_groupwise_metrics(...) and pickle the dict.
Then compare master vs variant pickles per (group, metric) entry.
"""

import argparse
import pickle
import shutil
import subprocess
import sys
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


def run_branch(branch_label, out_path):
    """Subprocess: load fork, build dataset+model, compute metrics, pickle."""
    script = """
import pickle, sys
import numpy as np
import torch

# Reset all caches.
for k in list(sys.modules):
    if k.startswith('declearn'):
        del sys.modules[k]

from declearn.fairness.core import FairnessInMemoryDataset
from declearn.fairness.api._metrics import FairnessMetricsComputer
from declearn.metrics import MeanMetric

# Pin torch RNG so model init is identical between branches.
torch.manual_seed(123)

# Build dataset from client_0 of data_iid_fair.
import pathlib
ROOT = pathlib.Path('/home/fslimani/declearn-bench/examples/mnist_quickrun/data_iid_fair/client_0')
data = np.load(ROOT / 'train_data.npy')
target = np.load(ROOT / 'train_target.npy')
s_attr = np.load(ROOT / 'train_s_attr.npy')

ds = FairnessInMemoryDataset(
    data=data, target=target, s_attr=s_attr, sensitive_target=True,
)

# Build a tiny torch model that maps 28*28 → 2 (binary).
import torch.nn as nn
from declearn.model.torch import TorchModel

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(28*28, 2)
    def forward(self, x):
        return self.fc(self.flatten(x))

network = TinyModel()
model = TorchModel(network, loss=nn.CrossEntropyLoss())

# Set up the metrics computer + an Accuracy metric (binary, thresh=0.5).
computer = FairnessMetricsComputer(ds)
accuracy = computer.setup_accuracy_metric(model, thresh=0.5)
metrics = [accuracy]

import time
# Warm-up to get past first-call overheads.
_ = computer.compute_groupwise_metrics(
    metrics=metrics, model=model, batch_size=128, n_batch=None,
)
t0 = time.perf_counter()
N_ITER = 5
for _ in range(N_ITER):
    result = computer.compute_groupwise_metrics(
        metrics=metrics, model=model, batch_size=128, n_batch=None,
    )
elapsed = (time.perf_counter() - t0) / N_ITER
result['_timing_per_call_s'] = elapsed
print(f'[{sys.argv[2]}] avg compute_groupwise_metrics call: {elapsed*1000:.1f} ms')
# Pickle: skip the timing entry for the entry-count reporter.
n_entries = sum(len(d) for k, d in result.items() if k != '_timing_per_call_s')

OUT = pathlib.Path(sys.argv[1])
OUT.write_bytes(pickle.dumps(result))
print(f'[{sys.argv[2]}] wrote {n_entries} entries -> {OUT}')
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

    print("=== exp_11 unit smoke ===\n")
    print("--- master ---")
    install("master")
    run_branch("master", out_master)

    print("\n--- variant ---")
    install("exp_11_fairness_singlepass_variant")
    run_branch("variant", out_variant)

    print("\n=== resetting venv ===")
    subprocess.run([PY, "-m", "pip", "install", "-e", str(REPO / "declearn"),
                    "-q"], check=True, capture_output=True)

    a = pickle.loads(out_master.read_bytes())
    b = pickle.loads(out_variant.read_bytes())
    assert a.keys() == b.keys(), f"metric-name mismatch: {a.keys()} vs {b.keys()}"

    failures = []
    rows = []
    # Pull and remove the timing entries before comparing.
    t_master = a.pop('_timing_per_call_s', None)
    t_variant = b.pop('_timing_per_call_s', None)
    print(f"\n{'metric':<10} {'group':<20} {'master':>14} {'variant':>14} {'abs_diff':>14}")
    for metric_name in sorted(a):
        m_groups = a[metric_name]
        v_groups = b[metric_name]
        if set(m_groups) != set(v_groups):
            failures.append((metric_name, "group set mismatch"))
            continue
        for g in m_groups:
            d = abs(m_groups[g] - v_groups[g])
            rows.append((metric_name, g, m_groups[g], v_groups[g], d))
            if d > 1e-6:
                failures.append((metric_name, f"group {g}: {d:.3e}"))
            print(f"{metric_name:<10} {str(g):<20} {m_groups[g]:>14.10f} {v_groups[g]:>14.10f} {d:>14.3e}")
    if t_master is not None and t_variant is not None:
        speedup = t_master / t_variant if t_variant else float('inf')
        print(f"\n--- compute_groupwise_metrics latency ---")
        print(f"master:  {t_master*1000:.1f} ms/call")
        print(f"variant: {t_variant*1000:.1f} ms/call")
        print(f"speedup: {speedup:.2f}×  ({(1 - t_variant/t_master)*100:.1f}% reduction)")

    if failures:
        print(f"\nFAIL: {len(failures)} mismatches above 1e-6")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print(f"\nPASS: {len(rows)} per-(metric, group) entries match "
          f"(max abs diff {max((d for *_, d in rows), default=0.0):.3e})")


if __name__ == "__main__":
    main()
