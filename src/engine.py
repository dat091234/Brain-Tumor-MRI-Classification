"""Training and validation loops with frozen BatchNorm handling and early stopping."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def freeze_bn_of_frozen_layers(model: nn.Module) -> None:
    """Keep BatchNorm2d layers in eval mode if all of their parameters are frozen."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if not any(param.requires_grad for param in module.parameters()):
                module.eval()


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch and return (epoch_loss, epoch_accuracy)."""
    model.train()
    freeze_bn_of_frozen_layers(model)

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on the validation set and return (val_loss, val_accuracy)."""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


def train_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    save_path: Union[str, Path],
    max_epochs: int = 30,
    patience: int = 10,
    initial_best_val_acc: float = 0.0,
) -> Tuple[float, List[Dict[str, float]]]:
    """Train a model for a given phase with early stopping and checkpoint saving.

    Args:
        model: Model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function (e.g., `nn.CrossEntropyLoss()`).
        optimizer: Configured PyTorch optimizer.
        device: Target compute device.
        save_path: File path to save the best model state_dict.
        max_epochs: Maximum number of epochs to train.
        patience: Number of epochs without validation accuracy improvement before stopping.
        initial_best_val_acc: Baseline validation accuracy from a previous phase.

    Returns:
        Tuple of (best_val_acc, history_list).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = initial_best_val_acc
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model -> {save_path} (Val Acc: {best_val_acc:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    return best_val_acc, history
