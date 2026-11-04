"""Dataset loading, stratified splitting, and image transformations."""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(
    normalize: bool = True,
    image_size: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """Build training, validation, and testing image transforms.

    Args:
        normalize: Whether to apply ImageNet mean/std normalization.
            Note: In the original experiments, Custom CNN uses unnormalized [0, 1]
            tensors (`normalize=False`), while pretrained models (ResNet18,
            DenseNet121, DINOv3) use ImageNet normalization (`normalize=True`).
        image_size: Target spatial resolution (height, width).

    Returns:
        Tuple of (train_transform, val_transform, test_transform).
    """
    
    if(normalize): norm_layers = [transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    else: norm_layers = []

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            *norm_layers,
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            *norm_layers,
        ]
    )

    return train_transform, eval_transform, eval_transform


def get_stratified_indices(
    train_dir: Union[str, Path],
    val_split: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate stratified training and validation indices from an ImageFolder directory."""
    full_dataset = ImageFolder(str(train_dir))
    targets = np.array(full_dataset.targets)
    indices = np.arange(len(full_dataset))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        random_state=random_state,
        stratify=targets,
    )
    return train_indices, val_indices


def create_datasets(
    train_dir: Union[str, Path],
    test_dir: Union[str, Path],
    normalize: bool = True,
    image_size: int = 224,
    val_split: float = 0.2,
    random_state: int = 42,
) -> Tuple[Subset, Subset, ImageFolder, List[str]]:
    """Create training, validation, and testing datasets with stratified split."""
    train_transform, val_transform, test_transform = get_transforms(
        normalize=normalize,
        image_size=image_size,
    )

    train_indices, val_indices = get_stratified_indices(
        train_dir=train_dir,
        val_split=val_split,
        random_state=random_state,
    )

    train_full = ImageFolder(str(train_dir), transform=train_transform)
    val_full = ImageFolder(str(train_dir), transform=val_transform)
    test_dataset = ImageFolder(str(test_dir), transform=test_transform)

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    return train_dataset, val_dataset, test_dataset, test_dataset.classes


def create_dataloaders(
    train_dir: Union[str, Path],
    test_dir: Union[str, Path],
    batch_size: int = 32,
    normalize: bool = True,
    image_size: int = 224,
    val_split: float = 0.2,
    random_state: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create PyTorch DataLoaders for training, validation, and testing."""
    train_dataset, val_dataset, test_dataset, classes = create_datasets(
        train_dir=train_dir,
        test_dir=test_dir,
        normalize=normalize,
        image_size=image_size,
        val_split=val_split,
        random_state=random_state,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, classes
