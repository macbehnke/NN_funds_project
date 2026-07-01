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


class ScaledTanh(nn.Module):
    """Scaled tanh activation used in LeCun et al. 1998."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.7159 * torch.tanh((2.0 / 3.0) * x)


class TrainableSubsampling(nn.Module):
    """LeNet-style average subsampling with one trainable scale and bias per map."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.activation = ScaledTanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return self.activation(x)


class PartialC3(nn.Module):
    """LeNet-5 C3 layer with the historical partial feature-map connectivity."""

    connection_table: tuple[tuple[int, ...], ...] = (
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4),
        (3, 4, 5),
        (0, 4, 5),
        (0, 1, 5),
        (0, 1, 2, 3),
        (1, 2, 3, 4),
        (2, 3, 4, 5),
        (0, 3, 4, 5),
        (0, 1, 4, 5),
        (0, 1, 2, 5),
        (0, 1, 3, 4),
        (1, 2, 4, 5),
        (0, 2, 3, 5),
        (0, 1, 2, 3, 4, 5),
    )

    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(len(inputs), 1, kernel_size=5) for inputs in self.connection_table
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for inputs, conv in zip(self.connection_table, self.convs):
            selected = x[:, inputs, :, :]
            outputs.append(conv(selected))
        return torch.cat(outputs, dim=1)


class LeNet5Faithful(nn.Module):
    """Historically faithful LeNet-5 architecture for 32x32 padded MNIST.

    This reproduces the layer sizes, scaled tanh activations, trainable
    subsampling layers, partial C3 connectivity, and convolutional C5 stage from
    LeCun et al. 1998. The final classifier is implemented as a modern linear
    84 -> 10 output trained with cross-entropy for stable PyTorch training.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.activation = ScaledTanh()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        self.s2 = TrainableSubsampling(6)
        self.c3 = PartialC3()
        self.s4 = TrainableSubsampling(16)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.c1(x))
        x = self.s2(x)
        x = self.activation(self.c3(x))
        x = self.s4(x)
        x = self.activation(self.c5(x))
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.f6(x))
        return self.output(x)


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
    if normalized == "lenet5_faithful":
        return LeNet5Faithful(num_classes=num_classes)
    if normalized == "resnet18":
        return build_resnet18_mnist(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def uses_32x32_input(name: str) -> bool:
    return name.lower() == "lenet5_faithful"


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
