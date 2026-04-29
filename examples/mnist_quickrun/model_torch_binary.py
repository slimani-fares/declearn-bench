"""Torch-backed binary classifier for the MNIST quickrun example.

Used by the fairness benchmarks, which binarize MNIST (digit<5 vs digit>=5)
so that all three fairness variants — fairbatch (binary-only),
fairfed, fairgrad — can share one setup.
"""

import torch

from declearn.model.torch import TorchModel

stack = [
    torch.nn.Unflatten(dim=0, unflattened_size=(-1, 1)),
    torch.nn.Conv2d(1, 8, 3, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Dropout(0.25),
    torch.nn.Flatten(),
    torch.nn.Linear(1352, 64),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(64, 2),
    torch.nn.Softmax(dim=-1),
]
network = torch.nn.Sequential(*stack)

model = TorchModel(network, loss=torch.nn.CrossEntropyLoss())
