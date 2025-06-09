import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from examples.utils import Accuracy

__all__ = ["train_step", "val_step", "test_step"]


def train_step(
    model: nn.Module,
    train_data: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    accuracy: Accuracy,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    r"""Perform a single training epoch.

    Args:
        model (nn.Module): The model to train.
        train_data (DataLoader): Training data loader.
        loss_fn (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        accuracy (Accuracy): Accuracy tracker.
        writer (SummaryWriter): TensorBoard writer.
        epoch (int): Current epoch number.
        device (torch.device): Device to run training on.
    """
    losses = []
    model.train()
    for inputs, targets in train_data:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs: torch.Tensor = model(inputs)
        loss_value = loss_fn(outputs, targets)
        loss_value.backward()
        optimizer.step()
        losses.append(loss_value.item())
        accuracy.update(outputs, targets)
    # Log the average training loss and accuracy for the epoch.
    writer.add_scalar("train/loss", torch.tensor(losses).mean().item(), epoch)
    writer.add_scalar("train/accuracy", accuracy.compute(), epoch)


def val_step(
    model: nn.Module,
    val_data: DataLoader,
    loss_fn: nn.Module,
    accuracy: Accuracy,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    r"""Perform a single validation epoch.

    Args:
        model (nn.Module): The model under evaluation.
        val_data (DataLoader): Validation data loader.
        loss_fn (nn.Module): Loss function.
        accuracy (Accuracy): Accuracy tracker.
        writer (SummaryWriter): TensorBoard writer.
        epoch (int): Current epoch number.
        device (torch.device): Device to run evaluation on.
    """
    losses = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_data:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs: torch.Tensor = model(inputs)
            loss_value = loss_fn(outputs, targets)
            losses.append(loss_value.item())
            accuracy.update(outputs, targets)
    writer.add_scalar("val/loss", torch.tensor(losses).mean().item(), epoch)
    writer.add_scalar("val/accuracy", accuracy.compute(), epoch)


def test_step(
    model: nn.Module,
    test_data: DataLoader,
    loss_fn: nn.Module,
    accuracy: Accuracy,
    device: torch.device,
) -> None:
    r"""Evaluate the model on the test dataset.

    Args:
        model (nn.Module): The model under evaluation.
        test_data (DataLoader): Test data loader.
        loss_fn (nn.Module): Loss function.
        accuracy (Accuracy): Accuracy tracker.
        device (torch.device): Device to run evaluation on.
    """
    losses = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_data:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs: torch.Tensor = model(inputs)
            loss_value = loss_fn(outputs, targets)
            losses.append(loss_value.item())
            accuracy.update(outputs, targets)
