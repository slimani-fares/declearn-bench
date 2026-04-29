"""Declearn model definition for haiku/jax MNIST benchmark.

Uses an MLP on flat (n, 784) MNIST input, matching the sklearn backend's
data layout. This keeps the haiku benchmark comparable to sklearn (both
linear/shallow on flat input) while exercising the jax/haiku code path.

Functions are defined at module level so declearn can pickle them for
serialization (see HaikuModel.get_config/from_config).
"""

import haiku as hk
import jax
import jax.numpy as jnp

from declearn.model.haiku import HaikuModel


def mnist_mlp_fn(inputs: jax.Array) -> jax.Array:
    """Simple MLP for MNIST: 784 -> 64 -> 10."""
    return hk.nets.MLP([64, 10])(inputs)


def cross_entropy_loss(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """Per-sample multi-class cross-entropy with integer class labels.

    y_pred is (batch, num_classes) logits; y_true is (batch,) integer labels
    in range [0, num_classes). Returns (batch,) per-sample loss values.
    """
    log_probs = jax.nn.log_softmax(y_pred, axis=-1)
    return -log_probs[jnp.arange(y_true.shape[0]), y_true.astype(jnp.int32)]


model = HaikuModel(mnist_mlp_fn, loss=cross_entropy_loss)