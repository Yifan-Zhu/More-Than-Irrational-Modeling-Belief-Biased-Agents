"""tests for the belief-biased user wrapper + the public API + a tiny training loop."""
import crci_mem
import gymnasium as gym
import pytest
from crci_mem.user_model.agent_forgetful import make_belief_wrapper


def test_top_level_api():
    assert hasattr(crci_mem, "make_belief_wrapper")
    assert hasattr(crci_mem, "NestedParticleFilter")
    assert crci_mem.__version__


@pytest.mark.parametrize("model", ["A", "B"])
def test_belief_wrapper_reset_step(model):
    wrapped = make_belief_wrapper(gym.make("MemoryDecayExp_9x9-v0"), 0.3, model)
    assert isinstance(wrapped.observation_space, gym.spaces.Dict)
    assert set(wrapped.observation_space.spaces.keys()) == {"position", "target"}
    obs, info = wrapped.reset(seed=0)
    assert set(obs.keys()) == {"position", "target"}
    obs, reward, terminated, truncated, info = wrapped.step(wrapped.action_space.sample())
    assert set(obs.keys()) == {"position", "target"}


@pytest.mark.slow
def test_tiny_ppo_train():
    from stable_baselines3 import PPO
    wrapped = make_belief_wrapper(gym.make("MemoryDecayExp_9x9-v0"), 0.0, "A")
    model = PPO("MultiInputPolicy", wrapped, n_steps=64, batch_size=64,
                n_epochs=1, device="cpu", seed=0, verbose=0)
    model.learn(total_timesteps=64)
