"""tests for the gym environments (import + reset/step contract)."""
import crci_mem  
import gymnasium as gym
import pytest

ENV_IDS = ["MemoryDecayExp_9x9-v0", "MemoryDecayMDP_9x9-v0"]


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_make_and_spaces(env_id):
    env = gym.make(env_id)
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert isinstance(env.action_space, gym.spaces.Discrete)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_reset_and_step(env_id):
    env = gym.make(env_id)
    obs, info = env.reset(seed=0)
    assert set(obs.keys()) == set(env.observation_space.spaces.keys())
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert set(obs.keys()) == set(env.observation_space.spaces.keys())
    assert float(reward) == reward  # reward is a scalar
