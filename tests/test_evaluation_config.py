from pathlib import Path

import numpy as np
import pytest

from dqn.config import TrainConfig
from dqn.evaluation import get_discounted_returns


def test_discounted_returns_with_bootstrap():
    returns = get_discounted_returns([1, 2, 3], gamma=0.5, bootstrap_value=4)
    assert np.allclose(returns, [3.25, 4.5, 5.0])


def test_all_json_configs_are_valid():
    config_paths = Path("configs").glob("*.json")
    configs = [TrainConfig.from_json(path) for path in config_paths]
    assert len(configs) >= 4
    assert any(config.env_name == "LunarLander-v3" for config in configs)
    assert any(config.double for config in configs)
    assert any(not config.double for config in configs)


def test_invalid_config_is_rejected():
    config = TrainConfig.from_json("configs/smoke_cartpole.json")
    values = config.to_dict()
    values["initial_fill"] = values["batch_size"] - 1
    with pytest.raises(ValueError):
        TrainConfig.from_dict(values)
