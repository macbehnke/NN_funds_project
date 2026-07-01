from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.optim import Adam, SGD
from tqdm import tqdm

from src.data import get_dataloaders
from src.model import build_loss, build_model, count_parameters, predictions_from_output, uses_32x32_input
from src.utils import ensure_dir, get_device, save_json, set_seed


def run_epoch(
    model: torch.nn.Module,
    model_name: str,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            output = model(images)
            loss = criterion(output, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (predictions_from_output(model_name, output) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train historical LeNet-5 or ResNet-18 on MNIST.")
    parser.add_argument(
        "--model",
        choices=["lenet5", "resnet18"],
        default="lenet5",
        help="Model architecture.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory for MNIST data.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Where to save model checkpoints.")
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/metrics.json"), help="Where to save final metrics.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument(
        "--optimizer",
        choices=["auto", "sgd", "adam"],
        default="auto",
        help="Optimizer. auto uses SGD for historical LeNet-5 and Adam for ResNet-18.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate. Defaults to 0.01 for SGD and 0.001 for Adam.",
    )
    parser.add_argument("--val-size", type=int, default=5000, help="Validation examples split from training set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count. Use 0 on Windows.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


def resolve_optimizer_name(model_name: str, optimizer_name: str) -> str:
    if optimizer_name != "auto":
        return optimizer_name
    if model_name == "lenet5":
        return "sgd"
    return "adam"


def default_learning_rate(optimizer_name: str) -> float:
    if optimizer_name == "sgd":
        return 1e-2
    return 1e-3


def build_optimizer(model: torch.nn.Module, optimizer_name: str, learning_rate: float) -> torch.optim.Optimizer:
    if optimizer_name == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
    if optimizer_name == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def serializable_args(args: argparse.Namespace) -> dict[str, str | int | float | bool | None]:
    clean_args: dict[str, str | int | float | bool | None] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            clean_args[key] = str(value)
        else:
            clean_args[key] = value
    return clean_args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)

    if args.model != "lenet5" and args.metrics_path == Path("outputs/metrics.json"):
        args.metrics_path = Path(f"outputs/{args.model}_metrics.json")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_size=args.val_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pad_to_32=uses_32x32_input(args.model),
    )

    model = build_model(args.model).to(device)
    criterion = build_loss(args.model)
    optimizer_name = resolve_optimizer_name(args.model, args.optimizer)
    learning_rate = args.lr if args.lr is not None else default_learning_rate(optimizer_name)
    optimizer = build_optimizer(model, optimizer_name, learning_rate)

    checkpoint_dir = ensure_dir(args.checkpoint_dir)
    best_checkpoint_path = checkpoint_dir / f"{args.model}_mnist_best.pt"

    best_val_accuracy = 0.0
    history: list[dict[str, float | int]] = []
    start = time.time()

    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Optimizer: {optimizer_name} lr={learning_rate}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, args.model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, args.model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "args": serializable_args(args),
                },
                best_checkpoint_path,
            )

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = run_epoch(model, args.model, test_loader, criterion, device)
    elapsed_seconds = time.time() - start

    metrics = {
        "model": args.model,
        "dataset": "MNIST",
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "parameters": count_parameters(model),
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "elapsed_seconds": elapsed_seconds,
        "history": history,
        "checkpoint": str(best_checkpoint_path),
    }
    save_json(metrics, args.metrics_path)

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Saved checkpoint: {best_checkpoint_path}")
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
