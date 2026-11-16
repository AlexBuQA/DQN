"""Сохранение и безопасная загрузка checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from .agents import make_agent
from .config import TrainConfig
from .utils import cpu_state_dict


def save_checkpoint(
    path: str | Path,
    agent,
    config: TrainConfig,
    step: int,
    validation: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "agent_type": config.agent,
        "state_shape": list(agent.state_shape),
        "n_actions": agent.n_actions,
        "hidden_dim": agent.hidden_dim,
        "model_state": cpu_state_dict(agent),
        "config": config.to_dict(),
        "step": int(step),
        "validation": validation,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device="cpu"):
    """Загружает только тензоры и простые данные, без pickle-объектов."""
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Совместимость со старыми версиями PyTorch.
        payload = torch.load(path, map_location=device)
    config = TrainConfig.from_dict(payload["config"])
    agent = make_agent(
        payload["agent_type"],
        payload["state_shape"],
        payload["n_actions"],
        epsilon=0.0,
        hidden_dim=payload["hidden_dim"],
    ).to(device)
    agent.load_state_dict(payload["model_state"])
    agent.eval()
    return agent, config, payload
