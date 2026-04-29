"""Declearn model definition for sklearn MNIST benchmark.

SklearnSGDModel is a linear model only (no CNN). For MNIST this becomes
a flat 784-input logistic regression classifier. The data must be
pre-reshaped from (n, 28, 28) to (n, 784)
"""

from declearn.model.sklearn import SklearnSGDModel

model = SklearnSGDModel.from_parameters(
    kind="classifier",
    loss="log_loss",
    penalty="l2",
    alpha=1e-4,
)