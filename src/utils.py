from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def relative_path(path: str | Path) -> str:
    try:
        return os.path.relpath(path, project_root())
    except ValueError:
        return str(path)
