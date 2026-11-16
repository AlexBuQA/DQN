"""Создание поддерживаемых сред Gymnasium."""

from __future__ import annotations

import gymnasium as gym


def make_env(env_name: str, render_mode: str | None = None):
    """Создаёт CartPole-v1 или дискретный LunarLander-v3."""
    if env_name == "LunarLander-v3":
        return gym.make(env_name, continuous=False, render_mode=render_mode)
    if env_name == "CartPole-v1":
        return gym.make(env_name, render_mode=render_mode)
    raise ValueError(f"Неподдерживаемая среда: {env_name}")
