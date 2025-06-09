import torch
import torch.nn as nn
import numpy as np

import os
import random

__all__ = ["Accuracy", "get_unique_filename", "set_seed", "count_parameters"]


class Accuracy:
    r"""Utility class to track classification accuracy over batches."""

    def __init__(self) -> None:
        self.correct: int = 0
        self.total: int = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        r"""Update counts based on model outputs and targets.

        Args:
            outputs (torch.Tensor): Model outputs (logits) of shape (batch, num_classes).
            targets (torch.Tensor): True class labels of shape (batch,).
        """
        preds: torch.Tensor = outputs.argmax(dim=1)
        self.correct += (preds == targets).sum().item()
        self.total += targets.size(0)

    def compute(self) -> float:
        r"""Compute the accuracy.

        Returns:
            float: Accuracy value.
        """
        return self.correct / self.total if self.total > 0 else 0

    def reset(self) -> None:
        r"""Reset the accuracy counts."""
        self.correct = 0
        self.total = 0


def get_unique_filename(filepath: str) -> str:
    r"""Generate a unique filename by appending an increasing index if the file exists.

    Args:
        filepath (str): Base filepath.

    Returns:
        str: Unique filepath that does not currently exist.
    """
    if not os.path.exists(filepath):
        return filepath

    base, ext = os.path.splitext(filepath)
    i = 1
    new_filepath = f"{base} ({i}){ext}"
    while os.path.exists(new_filepath):
        i += 1
        new_filepath = f"{base} ({i}){ext}"
    return new_filepath


def set_seed(seed: int) -> None:
    r"""Set seeds for reproducibility across numpy, random, and torch.

    Also configures PyTorch's deterministic algorithms.

    Args:
        seed (int): Seed value.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def count_parameters(model: nn.Module) -> int:
    r"""Count the number of trainable parameters in a model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: Total count of parameters with requires_grad=True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
