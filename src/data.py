from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def mnist_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    val_size: int,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_path = Path(data_dir)
    transform = mnist_transforms()

    full_train = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader
