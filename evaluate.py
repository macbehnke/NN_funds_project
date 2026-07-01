from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from src.data import get_dataloaders
from src.model import build_model, uses_32x32_input
from src.utils import get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained MNIST checkpoint.")
    parser.add_argument("--model", choices=["lenet5", "lenet5_faithful", "resnet18"], default="lenet5")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/lenet5_mnist_best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)

    if args.model != "lenet5" and args.checkpoint == Path("checkpoints/lenet5_mnist_best.pt"):
        args.checkpoint = Path(f"checkpoints/{args.model}_mnist_best.pt")
    if args.model != "lenet5" and args.output == Path("outputs/evaluation.json"):
        args.output = Path(f"outputs/{args.model}_evaluation.json")

    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_size=args.val_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pad_to_32=uses_32x32_input(args.model),
    )

    model = build_model(args.model).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            losses.append(loss.item() * labels.size(0))
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    accuracy = float((y_true_array == y_pred_array).mean())
    report = classification_report(y_true_array, y_pred_array, output_dict=True)
    matrix = confusion_matrix(y_true_array, y_pred_array).tolist()

    result = {
        "checkpoint": str(args.checkpoint),
        "test_loss": float(sum(losses) / len(y_true)),
        "test_accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
    }
    save_json(result, args.output)

    print(f"Device: {device}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Saved evaluation: {args.output}")


if __name__ == "__main__":
    main()
