from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from src.data import MNIST_MEAN, MNIST_STD
from src.model import LeNet5
from src.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create figures for the LeNet-5 MNIST report.")
    parser.add_argument("--metrics", type=Path, default=Path("outputs/metrics.json"))
    parser.add_argument("--evaluation", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/lenet5_mnist_best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def plot_training_curves(metrics: dict, output_dir: Path) -> None:
    history = metrics["history"]
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(9, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["train_accuracy"] for row in history], label="train")
    plt.plot(epochs, [row["val_accuracy"] for row in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close()


def plot_confusion_matrix(evaluation: dict, output_dir: Path) -> None:
    matrix = np.array(evaluation["confusion_matrix"])

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.title("MNIST Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.colorbar(fraction=0.046, pad=0.04)

    for row in range(10):
        for col in range(10):
            value = int(matrix[row, col])
            color = "white" if value > matrix.max() * 0.55 else "black"
            plt.text(col, row, value, ha="center", va="center", color=color, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close()


def plot_misclassified_examples(args: argparse.Namespace, output_dir: Path) -> None:
    device = get_device(prefer_cuda=not args.cpu)
    model = LeNet5().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )
    display_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    model_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(model_dataset, batch_size=256, shuffle=False, num_workers=args.num_workers)

    examples: list[tuple[int, int, int, float]] = []
    offset = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1).cpu()
            predictions = probabilities.argmax(dim=1)
            misses = predictions != labels
            miss_indices = torch.nonzero(misses).squeeze(1)

            for batch_index in miss_indices.tolist():
                true_label = int(labels[batch_index].item())
                predicted_label = int(predictions[batch_index].item())
                confidence = float(probabilities[batch_index, predicted_label].item())
                examples.append((offset + batch_index, true_label, predicted_label, confidence))
                if len(examples) >= 12:
                    break

            if len(examples) >= 12:
                break
            offset += labels.size(0)

    if not examples:
        print("No misclassified examples found.")
        return

    cols = 4
    rows = int(np.ceil(len(examples) / cols))
    plt.figure(figsize=(10, 2.6 * rows))

    for plot_index, (sample_index, true_label, predicted_label, confidence) in enumerate(examples, start=1):
        image, _ = display_dataset[sample_index]
        plt.subplot(rows, cols, plot_index)
        plt.imshow(image.squeeze(0), cmap="gray")
        plt.axis("off")
        plt.title(f"true {true_label}, pred {predicted_label}\nconf {confidence:.1%}")

    plt.tight_layout()
    plt.savefig(output_dir / "misclassified_examples.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))

    plot_training_curves(metrics, args.output_dir)
    plot_confusion_matrix(evaluation, args.output_dir)
    plot_misclassified_examples(args, args.output_dir)

    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
