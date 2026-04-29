# coding: utf-8

"""Torch-backed CNN model for the MNIST quickrun example, opacus-compatible.

Differs from model_torch.py by expecting channels-first input (B, 1, 28, 28)
directly, rather than using Unflatten(dim=0, ...) to inject a channel dim.
The Unflatten trick is incompatible with opacus's vmap-based per-sample
gradient mechanism, which hides the batch dimension from user-level layers.

Use with data prepared via _ensure_data_chw() in the benchmarks package.
"""

import torch

from declearn.model.torch import TorchModel

class FlexibleFlatten(torch.nn.Module):
    """Flatten that works under both vmap (no batch dim) and regular forward.
    
    Under vmap: input is (C, H, W), output is (C*H*W,)
    Regular:    input is (B, C, H, W), output is (B, C*H*W)
    """
    def forward(self, x):
        # If 4D, treat dim 0 as batch and preserve it.
        # If 3D (vmap stripped the batch), flatten everything.
        if x.dim() == 4:
            return x.flatten(start_dim=1)
        return x.flatten()

stack = [
    torch.nn.Conv2d(1, 8, 3, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Dropout(0.25),
    FlexibleFlatten(),
    torch.nn.Linear(1352, 64),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(64, 10),
    torch.nn.Softmax(dim=-1),
]
network = torch.nn.Sequential(*stack)

model = TorchModel(network, loss=torch.nn.CrossEntropyLoss())