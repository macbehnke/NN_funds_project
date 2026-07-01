from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from src.data import MNIST_MEAN, MNIST_STD


def normalize_mnist_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )
    return transform(image).unsqueeze(0)


def preprocess_user_file(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    image = ImageOps.invert(image)
    image = ImageOps.fit(image, (28, 28))
    return normalize_mnist_image(image)


def preprocess_canvas_data_url(data_url: str) -> torch.Tensor:
    """Convert a browser canvas PNG into a centered MNIST-style tensor.

    The browser canvas uses white strokes on a black background, so no color
    inversion is needed. The drawn digit is cropped, scaled to fit inside a
    20x20 box, and pasted into the center of a 28x28 image.
    """

    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    raw = base64.b64decode(data_url)
    image = Image.open(io.BytesIO(raw)).convert("L")
    array = np.array(image)
    mask = array > 15

    if not mask.any():
        blank = Image.new("L", (28, 28), color=0)
        return normalize_mnist_image(blank)

    ys, xs = np.where(mask)
    left, right = xs.min(), xs.max()
    top, bottom = ys.min(), ys.max()

    crop = image.crop((left, top, right + 1, bottom + 1))
    crop = ImageOps.pad(crop, (20, 20), method=Image.Resampling.LANCZOS, color=0, centering=(0.5, 0.5))

    mnist = Image.new("L", (28, 28), color=0)
    mnist.paste(crop, (4, 4))
    return normalize_mnist_image(mnist)
