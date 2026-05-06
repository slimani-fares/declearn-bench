"""Bigger Torch CNN for the exp_10 follow-up.

Designed to scale the parameter-tensor count (which is what
TorchVector dispatch optimizes) without exploding per-step compute.

Layout: shallow entry conv + 20 depth-20 conv+BN blocks at 8 channels +
small dense head. ~86 parameter tensors, ~50k scalar params.
"""

import torch

from declearn.model.torch import TorchModel

_DEPTH = 20
_CHANNELS = 8


class BiggerCNN(torch.nn.Module):
    def __init__(self, depth: int = _DEPTH, channels: int = _CHANNELS) -> None:
        super().__init__()
        self.entry = torch.nn.Sequential(
            torch.nn.Unflatten(dim=0, unflattened_size=(-1, 1)),
            torch.nn.Conv2d(1, channels, 3, padding=1),
            torch.nn.ReLU(),
        )
        blocks = []
        for _ in range(depth):
            blocks.append(torch.nn.Conv2d(channels, channels, 3, padding=1))
            blocks.append(torch.nn.BatchNorm2d(channels))
            blocks.append(torch.nn.ReLU())
        self.trunk = torch.nn.Sequential(*blocks)
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(channels, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.entry(x)))


network = BiggerCNN(depth=_DEPTH, channels=_CHANNELS)

# This needs to be called "model"; otherwise, a different name must be
# specified via the experiment's TOML configuration file.
model = TorchModel(network, loss=torch.nn.CrossEntropyLoss())
