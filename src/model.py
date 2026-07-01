from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class ScaledTanh(nn.Module):
    """Scaled tanh activation from LeCun et al. 1998."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.7159 * torch.tanh((2.0 / 3.0) * x)


class TrainableSubsampling(nn.Module):
    """LeNet-5 subsampling layer with trainable scale and bias per map."""

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
        return torch.cat([conv(x[:, inputs, :, :]) for inputs, conv in zip(self.connection_table, self.convs)], dim=1)


def digit_rbf_targets() -> torch.Tensor:
    """Fixed +1/-1 7x12 digit prototypes for the LeNet-5 RBF output layer."""

    digits = [
        ["0111110", "1100011", "1100011", "1100111", "1101111", "1111011", "1110011", "1100011", "1100011", "1100011", "0111110", "0000000"],
        ["0011000", "0111000", "1111000", "0011000", "0011000", "0011000", "0011000", "0011000", "0011000", "0011000", "1111110", "0000000"],
        ["0111110", "1100011", "0000011", "0000110", "0001100", "0011000", "0110000", "1100000", "1100000", "1100011", "1111111", "0000000"],
        ["0111110", "1100011", "0000011", "0000110", "0011110", "0000011", "0000011", "0000011", "0000011", "1100011", "0111110", "0000000"],
        ["0000110", "0001110", "0011110", "0110110", "1100110", "1100110", "1111111", "0000110", "0000110", "0000110", "0001111", "0000000"],
        ["1111111", "1100000", "1100000", "1100000", "1111110", "0000011", "0000011", "0000011", "0000011", "1100011", "0111110", "0000000"],
        ["0011110", "0110000", "1100000", "1100000", "1111110", "1100011", "1100011", "1100011", "1100011", "1100011", "0111110", "0000000"],
        ["1111111", "1100011", "0000011", "0000110", "0000110", "0001100", "0001100", "0011000", "0011000", "0011000", "0011000", "0000000"],
        ["0111110", "1100011", "1100011", "1100011", "0111110", "1100011", "1100011", "1100011", "1100011", "1100011", "0111110", "0000000"],
        ["0111110", "1100011", "1100011", "1100011", "1100011", "0111111", "0000011", "0000011", "0000011", "0000110", "0111100", "0000000"],
    ]
    rows = [[1.0 if char == "1" else -1.0 for row in digit for char in row] for digit in digits]
    return torch.tensor(rows, dtype=torch.float32)


class LeNet5(nn.Module):
    """Historical LeNet-5 reproduction from LeCun et al. 1998.

    The forward pass returns class penalties from fixed Euclidean RBF output
    units. Lower values indicate better class matches.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        if num_classes != 10:
            raise ValueError("Historical LeNet-5 RBF targets are defined for the 10 MNIST digit classes.")

        self.activation = ScaledTanh()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        self.s2 = TrainableSubsampling(6)
        self.c3 = PartialC3()
        self.s4 = TrainableSubsampling(16)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.f6 = nn.Linear(120, 84)
        self.register_buffer("rbf_targets", digit_rbf_targets())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.c1(x))
        x = self.s2(x)
        x = self.activation(self.c3(x))
        x = self.s4(x)
        x = self.activation(self.c5(x))
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.f6(x))
        return ((x.unsqueeze(1) - self.rbf_targets.unsqueeze(0)) ** 2).sum(dim=2)


class LeNet5MAPLoss(nn.Module):
    """Discriminative MAP-style loss from the LeNet-5 paper."""

    def __init__(self, rejection_penalty: float = 10.0) -> None:
        super().__init__()
        self.rejection_penalty = rejection_penalty

    def forward(self, penalties: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        correct = penalties.gather(1, labels.view(-1, 1)).squeeze(1)
        rejection = torch.full(
            (penalties.size(0), 1),
            -self.rejection_penalty,
            dtype=penalties.dtype,
            device=penalties.device,
        )
        competition = torch.logsumexp(torch.cat([rejection, -penalties], dim=1), dim=1)
        return (correct + competition).mean()


def build_resnet18_mnist(num_classes: int = 10) -> nn.Module:
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    if name.lower() == "lenet5":
        return LeNet5(num_classes=num_classes)
    if name.lower() == "resnet18":
        return build_resnet18_mnist(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def build_loss(name: str) -> nn.Module:
    if is_distance_output(name):
        return LeNet5MAPLoss()
    return nn.CrossEntropyLoss()


def is_distance_output(name: str) -> bool:
    return name.lower() == "lenet5"


def uses_32x32_input(name: str) -> bool:
    return name.lower() == "lenet5"


def predictions_from_output(name: str, output: torch.Tensor) -> torch.Tensor:
    if is_distance_output(name):
        return output.argmin(dim=1)
    return output.argmax(dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
