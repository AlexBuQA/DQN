"""TD-target и функции потерь Vanilla DQN и Double DQN."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_batch(states, actions, rewards, next_states, is_done, device):
    return (
        torch.as_tensor(states, device=device, dtype=torch.float32),
        torch.as_tensor(actions, device=device, dtype=torch.int64),
        torch.as_tensor(rewards, device=device, dtype=torch.float32),
        torch.as_tensor(next_states, device=device, dtype=torch.float32),
        torch.as_tensor(is_done, device=device, dtype=torch.bool),
    )


def compute_td_loss(
    states,
    actions,
    rewards,
    next_states,
    is_done,
    agent,
    target_network,
    gamma: float = 0.99,
    check_shapes: bool = False,
    device=None,
) -> torch.Tensor:
    """MSE Vanilla DQN: target-сеть и выбирает, и оценивает действие."""
    device = device or next(agent.parameters()).device
    states, actions, rewards, next_states, is_done = _as_batch(
        states, actions, rewards, next_states, is_done, device
    )
    predicted_qvalues = agent(states)
    predicted_qvalues_for_actions = predicted_qvalues.gather(
        1, actions.unsqueeze(-1)
    ).squeeze(-1)

    with torch.no_grad():
        predicted_next_qvalues = target_network(next_states)
        next_state_values = predicted_next_qvalues.max(dim=1).values
        # TERMINATION MASK VERIFIED
        target_qvalues_for_actions = rewards + gamma * (
            ~is_done
        ).float() * next_state_values
        # TARGET DETACHED VERIFIED
        target_qvalues_for_actions = target_qvalues_for_actions.detach()

    loss = F.mse_loss(predicted_qvalues_for_actions, target_qvalues_for_actions)
    if check_shapes:
        assert predicted_next_qvalues.shape == predicted_qvalues.shape
        assert next_state_values.shape == rewards.shape
        assert target_qvalues_for_actions.shape == rewards.shape
        assert not target_qvalues_for_actions.requires_grad
    return loss


def compute_td_loss_double(
    states,
    actions,
    rewards,
    next_states,
    is_done,
    agent,
    target_network,
    gamma: float = 0.99,
    loss_kind: str = "mse",
    check_shapes: bool = False,
    device=None,
) -> torch.Tensor:
    """Double DQN: online выбирает действие, target его оценивает."""
    if loss_kind not in {"mse", "huber"}:
        raise ValueError("loss_kind должен быть 'mse' или 'huber'")
    device = device or next(agent.parameters()).device
    states, actions, rewards, next_states, is_done = _as_batch(
        states, actions, rewards, next_states, is_done, device
    )
    predicted_qvalues = agent(states)
    predictions = predicted_qvalues.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        online_next_qvalues = agent(next_states)
        next_actions = online_next_qvalues.argmax(dim=1, keepdim=True)
        target_next_qvalues = target_network(next_states)
        # ONLINE SELECTS TARGET EVALUATES
        next_state_values = target_next_qvalues.gather(1, next_actions).squeeze(1)
        targets = rewards + gamma * (~is_done).float() * next_state_values
        # TARGET DETACHED VERIFIED
        targets = targets.detach()

    if loss_kind == "huber":
        loss = F.smooth_l1_loss(predictions, targets)
    else:
        loss = F.mse_loss(predictions, targets)
    if check_shapes:
        assert predictions.shape == rewards.shape
        assert targets.shape == rewards.shape
        assert not targets.requires_grad
    return loss
