from __future__ import annotations

import torch
from torch import nn


class LeNet5(nn.Module):
    """LeNet-5 style convolutional neural network for 28x28 MNIST digits.

    The original 1998 network used 32x32 inputs, tanh activations, average
    pooling, and a final Gaussian/RBF classifier. This implementation keeps the
    faithful small-CNN shape while using a standard linear classifier so it is
    simple, stable, and reproducible in modern PyTorch.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
