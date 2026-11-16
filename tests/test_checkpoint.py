import numpy as np

from dqn.agents import make_agent
from dqn.checkpoints import load_checkpoint, save_checkpoint
from dqn.config import TrainConfig


def test_checkpoint_round_trip(tmp_path):
    config = TrainConfig.from_json("configs/smoke_cartpole.json")
    agent = make_agent(config.agent, (4,), 2, hidden_dim=config.hidden_dim)
    path = tmp_path / "best.pt"
    save_checkpoint(path, agent, config, step=17, validation={"mean": 123.0})
    loaded, loaded_config, payload = load_checkpoint(path)
    assert loaded_config.name == config.name
    assert payload["step"] == 17
    states = np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
    assert np.allclose(agent.get_qvalues(states), loaded.get_qvalues(states))
