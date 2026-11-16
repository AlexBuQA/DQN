import numpy as np
import torch

from dqn.agents import DQNAgent, DuelingDQNAgent, DuelingDQNHead, make_agent


def test_dqn_shapes_and_gradients():
    agent = DQNAgent((4,), 2, hidden_dim=16)
    states = torch.randn(7, 4, requires_grad=True)
    qvalues = agent(states)
    assert qvalues.shape == (7, 2)
    qvalues.mean().backward()
    assert states.grad is not None
    assert all(p.grad is not None for p in agent.parameters())
    numpy_qvalues = agent.get_qvalues(np.zeros((3, 4), dtype=np.float32))
    assert numpy_qvalues.shape == (3, 2)


def test_epsilon_greedy_paths():
    np.random.seed(0)
    agent = DQNAgent((4,), 3, epsilon=0.0)
    qvalues = np.tile(np.array([[1.0, 5.0, 2.0]]), (100, 1))
    assert np.all(agent.sample_actions(qvalues) == 1)
    agent.epsilon = 1.0
    random_actions = agent.sample_actions(qvalues)
    assert np.any(random_actions != 1)
    assert np.all((0 <= random_actions) & (random_actions < 3))


def test_dueling_value_advantage_composition():
    torch.manual_seed(0)
    head = DuelingDQNHead(8, 3)
    features = torch.randn(5, 8)
    qvalues = head(features)
    value = head.value_stream(features).squeeze(1)
    assert qvalues.shape == (5, 3)
    assert torch.allclose(qvalues.mean(dim=1), value, atol=1e-6)


def test_dueling_agent_interface_and_factory():
    agent = make_agent("dueling", (4,), 2, epsilon=0.1, hidden_dim=16)
    assert isinstance(agent, DuelingDQNAgent)
    assert agent(torch.randn(6, 4)).shape == (6, 2)
    assert agent.get_qvalues(np.zeros((6, 4))).shape == (6, 2)
