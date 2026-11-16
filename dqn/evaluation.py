"""Независимая оценка политики и Monte-Carlo return."""

from __future__ import annotations

import numpy as np


def evaluate(
    env,
    agent,
    n_games: int = 1,
    greedy: bool = False,
    t_max: int = 500,
    seed: int | None = None,
    return_rewards: bool = False,
):
    """Оценивает средний reward на независимо инициализированных эпизодах."""
    rewards = []
    for game_idx in range(n_games):
        episode_seed = None if seed is None else seed + game_idx
        state, _ = env.reset(seed=episode_seed)
        total_reward = 0.0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([state])
            action = (
                int(qvalues.argmax(axis=-1)[0])
                if greedy
                else int(agent.sample_actions(qvalues)[0])
            )
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        rewards.append(total_reward)
    rewards_array = np.asarray(rewards, dtype=np.float32)
    return rewards_array if return_rewards else float(rewards_array.mean())


def play_and_log_episode(env, agent, seed=None, max_steps: int = 500) -> dict:
    """Записывает состояния, Q-значения, reward и признаки окончания."""
    state, _ = env.reset(seed=seed)
    log = {
        key: []
        for key in (
            "states", "qvalues", "actions", "rewards", "next_states",
            "terminated", "truncated",
        )
    }
    for _ in range(max_steps):
        qvalues = agent.get_qvalues([state])[0]
        action = int(np.argmax(qvalues))
        next_state, reward, terminated, truncated, _ = env.step(action)
        values = (state, qvalues, action, reward, next_state, terminated, truncated)
        for key, value in zip(log, values):
            log[key].append(value)
        state = next_state
        if terminated or truncated:
            break
    return {key: np.asarray(value) for key, value in log.items()}


def get_discounted_returns(
    rewards, gamma: float, bootstrap_value: float = 0.0
) -> np.ndarray:
    """Вычисляет G[t] справа налево с optional terminal bootstrap."""
    rewards_array = np.asarray(rewards, dtype=np.float64)
    returns = np.zeros_like(rewards_array, dtype=np.float64)
    running_return = float(bootstrap_value)
    for index in range(len(rewards_array) - 1, -1, -1):
        running_return = rewards_array[index] + gamma * running_return
        returns[index] = running_return
    return returns
