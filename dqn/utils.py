"""Воспроизводимость, устройства и сериализация результатов."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def linear_decay(start: float, final: float, step: int, decay_steps: int) -> float:
    fraction = min(max(step / decay_steps, 0.0), 1.0)
    return start + fraction * (final - start)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA запрошена, но недоступна")
    return device


def write_json(path: str | Path, data: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def cpu_state_dict(model: torch.nn.Module) -> dict:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}
