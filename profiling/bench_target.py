"""Subprocess target launched by run_pyspy.py.

Picks a benchmark class by name and runs its time_quickrun() method with
positional args in the order of its `param_names`. setup() is NOT called
here — the parent script must have called it before spawning.

Usage (called by the profiling runner, not directly):
    python profiling/bench_target.py --class <name> [--param <value>] [--n-clients <N>]
"""

import argparse
import os
import sys

# Ensure our local `benchmarks/` package shadows any same-named site-package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import (  # noqa: E402
    BackendsBenchmark,
    DPBenchmark,
    FairnessBenchmark,
    RegularizersBenchmark,
    ScaffoldBenchmark,
    SecAggBenchmark,
)

CLASSES = {
    "backends": BackendsBenchmark,
    "regularizers": RegularizersBenchmark,
    "scaffold": ScaffoldBenchmark,
    "dp": DPBenchmark,
    "secagg": SecAggBenchmark,
    "fairness": FairnessBenchmark,
}


def build_args(cls, param, n_clients):
    """Map CLI flags to positional args in the order of cls.param_names."""
    if not hasattr(cls, "param_names"):
        return ()
    mapping = {"config": param, "n_clients": n_clients}
    return tuple(mapping[name] for name in cls.param_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="class_name", required=True,
                    choices=sorted(CLASSES))
    ap.add_argument("--param", default=None,
                    help="TOML filename for classes whose param_names include 'config'.")
    ap.add_argument("--n-clients", dest="n_clients", default=None,
                    help="Client count for classes whose param_names include 'n_clients'.")
    args = ap.parse_args()

    cls = CLASSES[args.class_name]
    b = cls()
    positional = build_args(cls, args.param, args.n_clients)
    b.time_quickrun(*positional)


if __name__ == "__main__":
    main()
