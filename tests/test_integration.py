import numpy as np

from dqn.config import TrainConfig
from dqn.envs import make_env
from dqn.training import train


def test_cartpole_and_lunarlander_reset():
    for env_name, state_dim, n_actions in (
        ("CartPole-v1", 4, 2), ("LunarLander-v3", 8, 4)
    ):
        env = make_env(env_name)
        try:
            state, _ = env.reset(seed=0)
            assert state.shape == (state_dim,)
            next_state, reward, terminated, truncated, _ = env.step(0)
            assert next_state.shape == (state_dim,)
            assert np.isscalar(reward)
            assert isinstance(terminated, bool) and isinstance(truncated, bool)
            assert env.action_space.n == n_actions
        finally:
            env.close()


def test_tiny_training_pipeline(tmp_path):
    config = TrainConfig.from_json("configs/smoke_cartpole.json")
    config.name = "pytest_tiny"
    config.output_dir = str(tmp_path)
    config.replay_size = 64
    config.initial_fill = 32
    config.collect_steps = 16
    config.batch_size = 8
    config.total_steps = 2
    config.eval_freq = 1
    config.loss_freq = 1
    config.target_update_freq = 1
    config.validation_games = 2
    config.final_games = 2
    config.analysis_episodes = 0
    config.early_stopping = False
    summary = train(config, record_final_video=False)
    assert summary["executed_steps"] == 2
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "training_curves.png").is_file()
    assert (tmp_path / "final_scores.npy").is_file()
