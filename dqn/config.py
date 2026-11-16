"""Конфигурация воспроизводимого эксперимента."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class TrainConfig:
    name: str
    env_name: str
    agent: str
    double: bool
    loss_kind: str = "mse"
    seed: int = 42
    gamma: float = 0.99
    hidden_dim: int = 128
    max_episode_steps: int = 500
    replay_size: int = 10_000
    initial_fill: int = 10_000
    collect_steps: int = 100
    batch_size: int = 64
    total_steps: int = 60_000
    decay_steps: int = 25_000
    learning_rate: float = 5e-4
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.02
    target_update_freq: int = 500
    eval_freq: int = 2_000
    loss_freq: int = 100
    max_grad_norm: float = 10.0
    validation_games: int = 10
    validation_seed: int = 10_000
    validation_std_penalty: float = 0.0
    validation_reward_threshold: float = 490.0
    validation_p10_threshold: float = -1_000_000.0
    required_strong_evaluations: int = 2
    final_games: int = 30
    final_seed: int = 42_000
    final_threshold: float = 200.0
    analysis_episodes: int = 0
    analysis_seed: int = 250_001
    video_seed: int = 90_000
    output_dir: str = "runs/experiment"
    device: str = "auto"
    early_stopping: bool = True

    @classmethod
    def from_dict(cls, values: dict) -> "TrainConfig":
        known_fields = {field.name for field in fields(cls)}
        unknown = set(values) - known_fields
        if unknown:
            raise ValueError(f"Неизвестные параметры конфигурации: {sorted(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.agent not in {"dqn", "dueling"}:
            raise ValueError("agent должен быть 'dqn' или 'dueling'")
        if self.loss_kind not in {"mse", "huber"}:
            raise ValueError("loss_kind должен быть 'mse' или 'huber'")
        positive_fields = (
            "hidden_dim", "max_episode_steps", "replay_size", "initial_fill",
            "collect_steps", "batch_size", "total_steps", "decay_steps",
            "target_update_freq", "eval_freq", "loss_freq", "validation_games",
            "required_strong_evaluations", "final_games",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} должен быть положительным")
        if self.initial_fill < self.batch_size:
            raise ValueError("initial_fill должен быть не меньше batch_size")
        if self.initial_fill > self.replay_size:
            raise ValueError("initial_fill не должен превышать replay_size")
        if not 0.0 <= self.final_epsilon <= self.initial_epsilon <= 1.0:
            raise ValueError("Некорректный диапазон epsilon")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma должен находиться в диапазоне [0, 1]")
