"""MNIST data loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from mvae_mnist.config import DEFAULT_PATHS


class MNISTFactory(Protocol):
    def __call__(
        self,
        root: str,
        train: bool,
        download: bool,
        transform: object | None,
    ) -> Dataset:
        ...


@dataclass(frozen=True)
class MNISTDataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def build_mnist_dataloaders(
    data_dir: Path = DEFAULT_PATHS.data_dir,
    batch_size: int = 100,
    train_size: int = 50_000,
    val_size: int = 10_000,
    seed: int = 42,
    num_workers: int = 0,
    dataset_factory: MNISTFactory = MNIST,
) -> MNISTDataLoaders:
    """Create deterministic train/validation/test loaders for MNIST."""

    transform = transforms.ToTensor()
    train_dataset = dataset_factory(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = dataset_factory(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    requested = train_size + val_size
    if requested > len(train_dataset):
        raise ValueError(
            f"Requested {requested} train/validation examples, but MNIST train split "
            f"contains {len(train_dataset)} examples."
        )

    generator = torch.Generator().manual_seed(seed)
    remainder = len(train_dataset) - requested
    lengths = [train_size, val_size] + ([remainder] if remainder else [])
    subsets = random_split(train_dataset, lengths, generator=generator)
    train_subset, val_subset = subsets[0], subsets[1]

    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_subset,
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
    return MNISTDataLoaders(train=train_loader, val=val_loader, test=test_loader)
