"""ASV benchmarks for declearn across versions and configurations."""

import asyncio
import os
import shutil
import subprocess

import numpy as np

from declearn.quickrun._run import quickrun


# Paths anchored to the workspace root (two dirs up from this file).
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_ROOT = os.path.join(BENCH_ROOT, "configs")
DATA_FOLDER = os.path.join(BENCH_ROOT, "examples", "mnist_quickrun")

# Fixed seed so splits are identical across benchmark runs and declearn versions.
SPLIT_SEED = 42


def _data_folder_name(kind, n_clients):
    """Folder name convention: bare for n=3 (legacy), `_n{N}` suffix otherwise."""
    n_clients = int(n_clients)
    return f"data_{kind}" if n_clients == 3 else f"data_{kind}_n{n_clients}"


def _ensure_data(n_clients=3):
    """Split MNIST into examples/mnist_quickrun/data_iid[_n<N>] if not present.
    Uses a fixed seed for reproducibility across declearn versions."""
    import tempfile
    n_clients = int(n_clients)
    target_name = _data_folder_name("iid", n_clients)
    target = os.path.join(DATA_FOLDER, target_name)
    if os.path.isdir(target):
        return
    if n_clients == 3:
        # Legacy: declearn-split writes directly to <DATA_FOLDER>/data_iid/.
        subprocess.run(
            ["declearn-split", "--folder", DATA_FOLDER,
             "--seed", str(SPLIT_SEED)],
            check=True,
        )
        return
    # Non-default n_clients: split into a scratch dir, then move into place.
    # This avoids clobbering data_iid/ when it already exists from n=3.
    with tempfile.TemporaryDirectory(dir=DATA_FOLDER) as scratch:
        subprocess.run(
            ["declearn-split", "--folder", scratch,
             "--seed", str(SPLIT_SEED), "--n_shards", str(n_clients)],
            check=True,
        )
        os.rename(os.path.join(scratch, "data_iid"), target)


def _ensure_data_flat(n_clients=3):
    """Ensure a flat (n, 784) version of the MNIST split exists.

    SklearnSGDModel is a linear model and cannot consume (n, 28, 28) inputs.
    Rather than running declearn-split twice (which would produce different
    shuffles even with the same seed due to internal RNG state coupling),
    we reshape the existing data_iid/ split. This guarantees sklearn and
    torch/tensorflow benchmarks see identical sample assignments.

    Idempotent: if data_iid_flat[_n<N>]/ already exists, does nothing.
    """
    n_clients = int(n_clients)
    _ensure_data(n_clients)

    src_root = os.path.join(DATA_FOLDER, _data_folder_name("iid", n_clients))
    dst_root = os.path.join(DATA_FOLDER, _data_folder_name("iid_flat", n_clients))

    if os.path.isdir(dst_root):
        return

    os.makedirs(dst_root)

    for client_name in sorted(os.listdir(src_root)):
        src_client = os.path.join(src_root, client_name)
        dst_client = os.path.join(dst_root, client_name)
        if not os.path.isdir(src_client):
            continue
        os.makedirs(dst_client)

        # Reshape data arrays; copy target arrays unchanged.
        for split in ("train", "valid"):
            data_src = os.path.join(src_client, f"{split}_data.npy")
            target_src = os.path.join(src_client, f"{split}_target.npy")
            data_dst = os.path.join(dst_client, f"{split}_data.npy")
            target_dst = os.path.join(dst_client, f"{split}_target.npy")

            arr = np.load(data_src)
            np.save(data_dst, arr.reshape(arr.shape[0], -1))
            shutil.copy(target_src, target_dst)


def _ensure_data_chw(n_clients=3):
    """Ensure a channels-first (n, 1, 28, 28) version of the MNIST split exists.

    The default torch model uses `Unflatten(dim=0, ...)` to inject a channel
    dim, which breaks under opacus's vmap-based per-sample gradient mechanism
    (vmap hides dim 0, so the Unflatten splits the wrong axis).

    Rather than fix this in the model, we pre-shape the data to standard
    CNN input layout (N, 1, 28, 28) so the DP-compatible model file can
    drop the reshape layer entirely.

    Idempotent: if data_iid_chw[_n<N>]/ already exists, does nothing.
    """
    n_clients = int(n_clients)
    _ensure_data(n_clients)

    src_root = os.path.join(DATA_FOLDER, _data_folder_name("iid", n_clients))
    dst_root = os.path.join(DATA_FOLDER, _data_folder_name("iid_chw", n_clients))

    if os.path.isdir(dst_root):
        return

    os.makedirs(dst_root)

    for client_name in sorted(os.listdir(src_root)):
        src_client = os.path.join(src_root, client_name)
        dst_client = os.path.join(dst_root, client_name)
        if not os.path.isdir(src_client):
            continue
        os.makedirs(dst_client)

        for split in ("train", "valid"):
            data_src = os.path.join(src_client, f"{split}_data.npy")
            target_src = os.path.join(src_client, f"{split}_target.npy")
            data_dst = os.path.join(dst_client, f"{split}_data.npy")
            target_dst = os.path.join(dst_client, f"{split}_target.npy")

            arr = np.load(data_src)
            np.save(data_dst, arr.reshape(arr.shape[0], 1, 28, 28))
            # Cast targets to int64: CrossEntropyLoss under torch.func.vmap
            # dispatches through gather(), which rejects uint8 indices.
            tgt = np.load(target_src)
            np.save(target_dst, tgt.astype(np.int64))


def _patched_toml_for_clients(src_path, n_clients):
    """Return a TOML path with `data_folder` suffixed `_n<N>`, or src unchanged for n=3.

    Writes a sibling temp file next to the original (so os.path.dirname() stays
    under the project tree — checkpoints land alongside existing result_* dirs).
    Caller is responsible for deleting the returned temp path.
    """
    import re
    import tempfile
    n_clients = int(n_clients)
    if n_clients == 3:
        return src_path, None
    with open(src_path) as f:
        src = f.read()
    patched = re.sub(
        r'(data_folder\s*=\s*["\'])([^"\']+)(["\'])',
        lambda m: f"{m.group(1)}{m.group(2)}_n{n_clients}{m.group(3)}",
        src,
    )
    fd, tmp = tempfile.mkstemp(
        suffix=".toml", prefix="_tmp_n%d_" % n_clients,
        dir=os.path.dirname(src_path),
    )
    with os.fdopen(fd, "w") as f:
        f.write(patched)
    return tmp, tmp


def _run_quickrun(config_relpath, n_clients=3):
    """Run quickrun from BENCH_ROOT with the given config path."""
    config_path = os.path.join(CONFIGS_ROOT, config_relpath)
    toml_to_run, tmp = _patched_toml_for_clients(config_path, n_clients)
    original_cwd = os.getcwd()
    os.chdir(BENCH_ROOT)
    try:
        asyncio.run(quickrun(toml_to_run))
    finally:
        os.chdir(original_cwd)
        if tmp is not None and os.path.exists(tmp):
            os.remove(tmp)

class BackendsBenchmark:
    """Compare ML backends (torch, tensorflow, sklearn) under default FedAvg."""
    timeout = 300
    params = (
        [
            "config_fedavg_torch.toml",
            "config_fedavg_tensorflow.toml",
            # "config_fedavg_sklearn.toml", excluded: takes too long.
        ],
        [3, 5],
    )
    param_names = ["config", "n_clients"]

    def setup(self, config, n_clients):
        _ensure_data(n_clients)
        _ensure_data_flat(n_clients)

    def time_quickrun(self, config, n_clients):
        _run_quickrun(f"backends/{config}", n_clients=n_clients)


class RegularizersBenchmark:
    """Compare regularizers (lasso, ridge, fedprox) under torch FedAvg."""
    timeout = 300
    params = (
        [
            "config_fedavg_torch_lasso.toml",
            "config_fedavg_torch_ridge.toml",
            "config_fedavg_torch_fedprox.toml",
        ],
        [3, 5],
    )
    param_names = ["config", "n_clients"]

    def setup(self, config, n_clients):
        _ensure_data(n_clients)

    def time_quickrun(self, config, n_clients):
        _run_quickrun(f"regularizers/{config}", n_clients=n_clients)


class ScaffoldBenchmark:
    """SCAFFOLD algorithm with torch backend, swept over client counts."""
    timeout = 300
    params = [3, 5]
    param_names = ["n_clients"]

    def setup(self, n_clients):
        _ensure_data(n_clients)

    def time_quickrun(self, n_clients):
        _run_quickrun(
            "scaffold/config_fedavg_torch_scaffold.toml",
            n_clients=n_clients,
        )


class DPBenchmark:
    """FedAvg with DP-SGD (opacus) under torch backend, swept over client counts."""
    timeout = 600
    params = [3, 5]
    param_names = ["n_clients"]

    def setup(self, n_clients):
        _ensure_data_chw(n_clients)

    def time_quickrun(self, n_clients):
        _run_quickrun("dp/config_fedavg_torch_dp.toml", n_clients=n_clients)

