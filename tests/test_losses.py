import numpy as np
import pytest
import torch
from torch import nn

from dqn.agents import DQNAgent
from dqn.losses import compute_td_loss, compute_td_loss_double


class TableQ(nn.Module):
    def __init__(self, table):
        super().__init__()
        self.table = nn.Parameter(torch.tensor(table, dtype=torch.float32), requires_grad=False)

    def forward(self, states):
        indices = states.long().reshape(-1)
        return self.table[indices]


def test_vanilla_dqn_exact_target_and_terminal_mask():
    agent = TableQ([[1.0, 2.0], [3.0, 4.0]])
    target = TableQ([[3.0, 10.0], [7.0, 8.0]])
    loss = compute_td_loss(
        states=[[0], [1]], actions=[0, 1], rewards=[1.0, 2.0],
        next_states=[[0], [1]], is_done=[False, True],
        agent=agent, target_network=target, gamma=0.5,
    )
    # targets = [6, 2], predictions = [1, 4], MSE = (25 + 4) / 2.
    assert loss.item() == pytest.approx(14.5)


def test_double_dqn_online_selects_target_evaluates():
    agent = TableQ([[1.0, 2.0], [3.0, 4.0], [1.0, 5.0], [4.0, 2.0]])
    target = TableQ([[9.0, 9.0], [9.0, 9.0], [10.0, 20.0], [30.0, 40.0]])
    loss = compute_td_loss_double(
        states=[[0], [1]], actions=[0, 1], rewards=[1.0, 2.0],
        next_states=[[2], [3]], is_done=[False, True],
        agent=agent, target_network=target, gamma=0.5,
    )
    # Online выбирает действия [1, 0], target даёт [20, 30].
    # targets = [11, 2], predictions = [1, 4], MSE = 52.
    assert loss.item() == pytest.approx(52.0)


def test_target_network_receives_no_gradients():
    agent = DQNAgent((4,), 2, hidden_dim=16)
    target = DQNAgent((4,), 2, hidden_dim=16)
    target.load_state_dict(agent.state_dict())
    batch = (
        np.random.randn(8, 4).astype(np.float32),
        np.random.randint(0, 2, size=8),
        np.random.randn(8).astype(np.float32),
        np.random.randn(8, 4).astype(np.float32),
        np.random.randint(0, 2, size=8).astype(bool),
    )
    loss = compute_td_loss_double(*batch, agent, target, check_shapes=True)
    loss.backward()
    assert loss.requires_grad and loss.ndim == 0
    assert any(p.grad is not None for p in agent.parameters())
    assert all(p.grad is None for p in target.parameters())


def test_huber_loss_and_invalid_loss_kind():
    agent = TableQ([[1.0, 2.0]])
    target = TableQ([[3.0, 4.0]])
    args = ([[0]], [0], [1.0], [[0]], [False])
    huber = compute_td_loss_double(*args, agent, target, loss_kind="huber")
    assert torch.isfinite(huber)
    with pytest.raises(ValueError):
        compute_td_loss_double(*args, agent, target, loss_kind="unknown")
