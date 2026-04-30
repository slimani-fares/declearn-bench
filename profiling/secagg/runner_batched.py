"""SecAgg profiling target. Called by run_pyspy_secagg.py under py-spy."""
import argparse
import asyncio
import os
import tempfile

import tomli
import tomli_w

from declearn.quickrun._run import quickrun


def patch_toml(config_path: str, n_clients: int) -> str:
    """Inject min_clients=n_clients into the TOML, write to temp file."""
    with open(config_path, "rb") as f:
        cfg = tomli.load(f)
    cfg.setdefault("run", {}).setdefault("register", {})["min_clients"] = n_clients
    fd, tmp = tempfile.mkstemp(suffix=".toml", prefix="secagg_")
    os.close(fd)
    with open(tmp, "wb") as f:
        tomli_w.dump(cfg, f)
    return tmp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n_clients", type=int, required=True)
    args = parser.parse_args()

    tmp_toml = patch_toml(args.config, args.n_clients)
    try:
        asyncio.run(quickrun(tmp_toml, secagg_variant="masking"))
    finally:
        os.remove(tmp_toml)


if __name__ == "__main__":
    main()
