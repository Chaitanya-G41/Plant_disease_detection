"""
Dataset and DataLoader construction.
Handles class weighting for imbalanced PlantVillage classes.
Separates guava classes for targeted evaluation.
"""

import os
import numpy as np
from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from src.preprocess import get_transforms


# Guava class prefix — used to identify target classes
GUAVA_PREFIX = 'Guava_'


def get_class_weights(dataset: ImageFolder) -> torch.Tensor:
    """
    Compute inverse-frequency class weights.
    Prevents Orange (4415 images) from dominating over
    Potato_healthy (120 images) during Stage 1 training.
    """
    counts = Counter(dataset.targets)
    total  = sum(counts.values())
    n_cls  = len(counts)

    # Weight = total / (n_classes * count_for_class)
    weights = torch.zeros(n_cls)
    for cls_idx, count in counts.items():
        weights[cls_idx] = total / (n_cls * count)

    return weights


def get_sample_weights(dataset: ImageFolder) -> torch.Tensor:
    """
    Per-sample weights for WeightedRandomSampler.
    Each sample gets the weight of its class.
    """
    class_weights = get_class_weights(dataset)
    sample_weights = torch.tensor(
        [class_weights[t] for t in dataset.targets],
        dtype=torch.float
    )
    return sample_weights


def get_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
    use_weighted_sampler: bool = True
):
    """
    Build train and val dataloaders.

    Args:
        train_dir            : path to train folder
        val_dir              : path to val folder
        batch_size           : images per batch
        num_workers          : parallel workers (2 works well on Colab)
        use_weighted_sampler : if True, balances class sampling during training

    Returns:
        train_loader, val_loader, class_names, idx_to_class
    """
    train_dataset = ImageFolder(root=train_dir, transform=get_transforms('train'))
    val_dataset   = ImageFolder(root=val_dir,   transform=get_transforms('val'))

    # Weighted sampler — balances batches across 44 uneven classes
    if use_weighted_sampler:
        sample_weights = get_sample_weights(train_dataset)
        sampler = WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = len(sample_weights),
            replacement = True
        )
        shuffle = False  # mutually exclusive with sampler
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        sampler     = sampler,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True   # avoids batch norm issues on last small batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True
    )

    class_names  = train_dataset.classes
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    # Print summary
    guava_classes = [c for c in class_names if c.startswith(GUAVA_PREFIX)]
    other_classes = [c for c in class_names if not c.startswith(GUAVA_PREFIX)]

    print(f"Total classes     : {len(class_names)}")
    print(f"PlantVillage      : {len(other_classes)}")
    print(f"Guava (target)    : {len(guava_classes)} → {guava_classes}")
    print(f"Train images      : {len(train_dataset)}")
    print(f"Val images        : {len(val_dataset)}")
    print(f"Batch size        : {batch_size}")
    print(f"Weighted sampler  : {use_weighted_sampler}")

    return train_loader, val_loader, class_names, idx_to_class


def get_guava_indices(class_names: list) -> list:
    """
    Returns indices of guava classes.
    Used during Stage 2 and evaluation to isolate guava performance.
    """
    return [i for i, c in enumerate(class_names) if c.startswith(GUAVA_PREFIX)]