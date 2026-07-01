from __future__ import annotations

import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from src.data import MNIST_MEAN, MNIST_STD
from src.model import build_model, predictions_from_output
from src.preprocess import preprocess_user_file
from src.utils import get_device


def preprocess_user_image(path: Path) -> torch.Tensor:
    return preprocess_user_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LeNet-5 MNIST demo predictions.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/lenet5_mnist_best.pt"))
    parser.add_argument("--image", type=Path, default=None, help="Optional path to a handwritten digit image.")
    parser.add_argument("--sample-index", type=int, default=0, help="MNIST test sample index used when --image is omitted.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-figure", type=Path, default=Path("outputs/demo_prediction.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(prefer_cuda=not args.cpu)

    model = build_model("lenet5").to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.image is not None:
        image_tensor = preprocess_user_image(args.image)
        title_source = str(args.image)
        display_image = image_tensor.squeeze(0).squeeze(0).numpy()
        true_label = None
    else:
        dataset = datasets.MNIST(
            root=args.data_dir,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Pad(2),
                    transforms.ToTensor(),
                    transforms.Normalize(MNIST_MEAN, MNIST_STD),
                ]
            ),
        )
        image_tensor, true_label = dataset[args.sample_index]
        image_tensor = image_tensor.unsqueeze(0)
        title_source = f"MNIST test sample #{args.sample_index}"
        display_image = image_tensor.squeeze(0).squeeze(0).numpy()

    with torch.no_grad():
        output = model(image_tensor.to(device))
        probabilities = torch.softmax(-output, dim=1).squeeze(0).cpu()
        prediction = int(predictions_from_output("lenet5", output).item())
        confidence = float(probabilities[prediction].item())

    args.save_figure.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(display_image, cmap="gray")
    plt.axis("off")
    if true_label is None:
        plt.title(f"Prediction: {prediction} ({confidence:.1%})")
    else:
        plt.title(f"True: {true_label} | Prediction: {prediction} ({confidence:.1%})")
    plt.tight_layout()
    plt.savefig(args.save_figure, dpi=160)

    print(f"Source: {title_source}")
    if true_label is not None:
        print(f"True label: {true_label}")
    print(f"Predicted label: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Saved figure: {args.save_figure}")


if __name__ == "__main__":
    main()
