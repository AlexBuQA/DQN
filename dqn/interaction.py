"""Взаимодействие агента со средой и запись переходов."""

from __future__ import annotations

from .agents import DQNAgent
from .replay import ReplayBuffer


def play_and_record(
    initial_state,
    agent: DQNAgent,
    env,
    exp_replay: ReplayBuffer,
    n_steps: int = 1,
):
    """Выполняет ровно ``n_steps`` шагов и возвращает reward и состояние.

    В replay ``done`` означает только ``terminated``. При ``truncated``
    сохраняется bootstrap через next_state, но эпизод всё равно перезапускается.
    """
    if n_steps < 1:
        raise ValueError("n_steps должен быть положительным")
    state = initial_state
    sum_rewards = 0.0

    for _ in range(n_steps):
        qvalues = agent.get_qvalues([state])
        action = int(agent.sample_actions(qvalues)[0])
        next_state, reward, terminated, truncated, _ = env.step(action)
        # TERMINATED IS NOT TRUNCATED
        exp_replay.add(state, action, reward, next_state, bool(terminated))
        sum_rewards += float(reward)

        if terminated or truncated:
            # TRUNCATION KEEPS BOOTSTRAP
            # EPISODE RESET VERIFIED
            state, _ = env.reset()
        else:
            state = next_state

    return sum_rewards, state
