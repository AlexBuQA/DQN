import numpy as np
import pytest

from dqn.interaction import play_and_record
from dqn.replay import ReplayBuffer


class ZeroAgent:
    def get_qvalues(self, states):
        return np.zeros((len(states), 2), dtype=np.float32)

    def sample_actions(self, qvalues):
        return np.zeros(len(qvalues), dtype=np.int64)


class TerminationProbeEnv:
    """Шаг 2 — truncated, шаг 3 после reset — terminated."""

    def __init__(self):
        self.steps = 0
        self.resets = 0

    def reset(self, seed=None):
        self.resets += 1
        return np.array([10.0 + self.resets], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        if self.steps == 1:
            return np.array([2.0], dtype=np.float32), 1.0, False, False, {}
        if self.steps == 2:
            return np.array([3.0], dtype=np.float32), 2.0, False, True, {}
        return np.array([4.0], dtype=np.float32), 3.0, True, False, {}


def test_replay_is_circular_and_returns_typed_batches():
    replay = ReplayBuffer(3)
    for index in range(5):
        replay.add([index], index % 2, float(index), [index + 1], index == 4)
    assert len(replay) == 3
    stored_rewards = [item[2] for item in replay._storage]
    assert stored_rewards == [3.0, 4.0, 2.0]
    batch = replay.sample(20)
    assert batch[0].shape == (20, 1)
    assert batch[1].dtype == np.int64
    assert batch[2].dtype == np.float32
    assert batch[4].dtype == np.bool_
    assert set(np.unique(batch[2])) <= {2.0, 3.0, 4.0}


def test_truncation_keeps_bootstrap_but_resets_episode():
    env = TerminationProbeEnv()
    replay = ReplayBuffer(10)
    initial_state, _ = env.reset()
    reward, final_state = play_and_record(
        initial_state, ZeroAgent(), env, replay, n_steps=3
    )
    assert reward == pytest.approx(6.0)
    assert len(replay) == 3
    assert [item[4] for item in replay._storage] == [False, False, True]
    assert replay._storage[1][3].item() == 3.0  # next_state truncated сохранён
    assert env.resets == 3  # начальный reset + truncated + terminated
    assert final_state.item() == pytest.approx(13.0)


def test_empty_replay_sampling_is_rejected():
    with pytest.raises(ValueError):
        ReplayBuffer(2).sample(1)
