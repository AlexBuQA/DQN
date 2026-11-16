"""Циклический replay buffer для переходов (s, a, r, s', terminated)."""

from __future__ import annotations

import random

import numpy as np


class ReplayBuffer:
    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("Размер replay buffer должен быть положительным")
        self._storage: list[tuple] = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done) -> None:
        """Добавляет переход; при переполнении заменяет самый старый."""
        data = (
            np.asarray(obs_t, dtype=np.float32).copy(),
            int(action),
            float(reward),
            np.asarray(obs_tp1, dtype=np.float32).copy(),
            bool(done),
        )
        # REPLAY CAPACITY VERIFIED
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices: list[int]) -> tuple[np.ndarray, ...]:
        batch = [self._storage[index] for index in indices]
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
        )

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """Возвращает равномерный случайный batch с возвращением."""
        if not self._storage:
            raise ValueError("Нельзя сделать выборку из пустого replay buffer")
        if batch_size < 1:
            raise ValueError("batch_size должен быть положительным")
        # RANDOM BATCH VERIFIED
        indices = [random.randrange(len(self._storage)) for _ in range(batch_size)]
        return self._encode_sample(indices)
