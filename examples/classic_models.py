import torch
import torch.nn as nn

import math

__all__ = ["ParamMatchedFCN"]


class ParamMatchedFCN(nn.Module):
    r"""Single-hidden-layer MLP with a *fixed* minimum number of parameters *d*.

    Ensures that the total number of weight parameters is ≥ *d* and as close to *d* as possible.
    If *d* exactly equals in_features · out_features, uses a single Linear layer.
    Otherwise, adds one hidden relu layer of size
        ceil(d / (in_features + out_features))
    so that
        in_features*hidden + hidden*out_features ≥ d.
    """

    def __init__(self, in_features: int, out_features: int, d: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.d = d

        # Compute minimal hidden size h so that
        # in_features*h + h*out_features ≥ d
        hidden = max(1, math.floor(d / (in_features + out_features)))
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, out_features, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
