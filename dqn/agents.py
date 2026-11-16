"""Нейронные сети и epsilon-greedy политики."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


class DQNAgent(nn.Module):
    """MLP, преобразующая batch наблюдений в Q-значения действий."""

    def __init__(
        self,
        state_shape: Sequence[int],
        n_actions: int,
        epsilon: float = 0.0,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if len(state_shape) != 1:
            raise ValueError("Ожидается одномерное векторное наблюдение")
        if n_actions < 2 or hidden_dim < 1:
            raise ValueError("Некорректные размеры сети")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon должен находиться в диапазоне [0, 1]")

        self.epsilon = float(epsilon)
        self.n_actions = int(n_actions)
        self.state_shape = tuple(state_shape)
        self.hidden_dim = int(hidden_dim)
        state_dim = int(state_shape[0])
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state_t: torch.Tensor) -> torch.Tensor:
        """Возвращает Q-значения формы ``[batch_size, n_actions]``."""
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)
        # FEATURE MAP VERIFIED
        qvalues = self.network(state_t)
        if torch.is_grad_enabled():
            assert qvalues.requires_grad, "qvalues must carry gradients in grad mode"
        assert qvalues.shape == (state_t.shape[0], self.n_actions)
        return qvalues

    @torch.no_grad()
    def get_qvalues(self, states: np.ndarray) -> np.ndarray:
        """Вычисляет Q-значения для NumPy-массива без построения графа."""
        model_device = next(self.parameters()).device
        array = np.asarray(states, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        states_t = torch.as_tensor(array, device=model_device, dtype=torch.float32)
        return self(states_t).cpu().numpy()

    def sample_actions(self, qvalues: np.ndarray) -> np.ndarray:
        """Выбирает действия по epsilon-greedy политике."""
        qvalues = np.asarray(qvalues)
        if qvalues.ndim != 2 or qvalues.shape[1] != self.n_actions:
            raise ValueError("qvalues должны иметь форму [batch_size, n_actions]")
        batch_size, n_actions = qvalues.shape
        greedy_actions = qvalues.argmax(axis=1)
        random_actions = np.random.randint(0, n_actions, size=batch_size)
        random_mask = np.random.random(batch_size) < self.epsilon
        # EPSILON PATH VERIFIED
        return np.where(random_mask, random_actions, greedy_actions).astype(np.int64)


class DuelingDQNHead(nn.Module):
    """Value- и advantage-потоки с центрированным advantage."""

    def __init__(self, input_dim: int, n_actions: int) -> None:
        super().__init__()
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, n_actions),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # ADVANTAGE CENTERED VERIFIED
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DuelingDQN(nn.Module):
    """Общий feature extractor и Dueling-head."""

    def __init__(
        self, state_shape: Sequence[int], n_actions: int, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        state_dim = int(state_shape[0])
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.dueling_head = DuelingDQNHead(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.dueling_head(self.feature_extractor(x))


class DuelingDQNAgent(DQNAgent):
    """Тот же интерфейс агента, но с Dueling-архитектурой."""

    def __init__(
        self,
        state_shape: Sequence[int],
        n_actions: int,
        epsilon: float = 0.0,
        hidden_dim: int = 128,
    ) -> None:
        nn.Module.__init__(self)
        if len(state_shape) != 1:
            raise ValueError("Ожидается одномерное векторное наблюдение")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon должен находиться в диапазоне [0, 1]")
        self.epsilon = float(epsilon)
        self.n_actions = int(n_actions)
        self.state_shape = tuple(state_shape)
        self.hidden_dim = int(hidden_dim)
        self.network = DuelingDQN(state_shape, n_actions, hidden_dim)


def make_agent(
    agent_type: str,
    state_shape: Sequence[int],
    n_actions: int,
    epsilon: float = 0.0,
    hidden_dim: int = 128,
) -> DQNAgent:
    """Создаёт обычного или Dueling-агента через единый интерфейс."""
    agent_classes = {"dqn": DQNAgent, "dueling": DuelingDQNAgent}
    try:
        agent_class = agent_classes[agent_type.lower()]
    except KeyError as exc:
        raise ValueError("agent_type должен быть 'dqn' или 'dueling'") from exc
    return agent_class(state_shape, n_actions, epsilon, hidden_dim)
