"""DQN, Double DQN и Dueling DQN для дискретных сред Gymnasium."""

from .agents import DQNAgent, DuelingDQN, DuelingDQNAgent, DuelingDQNHead, make_agent
from .config import TrainConfig
from .evaluation import evaluate, get_discounted_returns, play_and_log_episode
from .interaction import play_and_record
from .losses import compute_td_loss, compute_td_loss_double
from .replay import ReplayBuffer

__all__ = [
    "DQNAgent",
    "DuelingDQN",
    "DuelingDQNAgent",
    "DuelingDQNHead",
    "ReplayBuffer",
    "TrainConfig",
    "compute_td_loss",
    "compute_td_loss_double",
    "evaluate",
    "get_discounted_returns",
    "make_agent",
    "play_and_log_episode",
    "play_and_record",
]
