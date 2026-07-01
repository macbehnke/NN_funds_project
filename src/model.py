from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


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


def build_resnet18_mnist(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for small 1-channel MNIST images.

    This is the "what we would do today" baseline. It keeps the ResNet-18
    residual block design, but changes the image stem so 28x28 MNIST images are
    not aggressively downsampled at the first layer.
    """

    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    normalized = name.lower()
    if normalized == "lenet5":
        return LeNet5(num_classes=num_classes)
    if normalized == "resnet18":
        return build_resnet18_mnist(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
