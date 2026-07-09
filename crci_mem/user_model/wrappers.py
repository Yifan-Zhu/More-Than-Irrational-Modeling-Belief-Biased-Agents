import copy
import gymnasium as gym
import torch as th
from torch import nn
import numpy as np
from crci_mem.envs.core import Belief, memory_decay_rand_drop, memory_decay_perfect
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def memory_decay_rand_drop_with_hints(history: list, decay: float, **kwargs):
    """
    Memory on target observation will be dropped with probability decay * 100%, a new memory hint is added additionally.
    """
    new_history = []
    for hist in history:
        is_hint = hist.get("hint", False)
        if is_hint or np.random.rand() >= decay:
            new_history.append(hist)
        else:
            new_hist = copy.deepcopy(hist)
            new_hist["obs"]["target"] = 0
            new_history.append(new_hist)
    return new_history

memory_model = {
    'perfect': memory_decay_perfect,
    'rand_drop_wo_hints': memory_decay_rand_drop,
    'rand_drop': memory_decay_rand_drop_with_hints,
}

class CombinedExtractor_Obs(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
            elif key == "surrounding":
                extractors[key] = nn.Sequential(
                    nn.Flatten(),  
                    nn.Linear(np.prod(subspace.shape), 64),  
                    nn.ReLU(),
                    nn.Linear(64, 16),  
                    nn.ReLU()
                )
                total_concat_size += 16  
            elif key == "target":
                extractors[key] = nn.Sequential(
                    nn.Embedding(subspace.n, 8),
                    nn.Flatten(),
                )
                total_concat_size += subspace.n * 8

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            if key == "target":
                target = observations[key]
                target = target.long()
                encoded_tensor_list.append(extractor(target))
            else:
                encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)
    
class CombinedExtractor_Belief(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "position":
                extractors[key] = nn.Sequential(
                    nn.Flatten(),  
                    nn.Linear(np.prod(subspace.shape), 32),  
                    nn.ReLU(),
                    nn.Linear(32, 16),  
                    nn.ReLU()
                )
                total_concat_size += 16  
            elif key == "target":
                extractors[key] = nn.Flatten()
                total_concat_size += np.prod(subspace.shape)
            elif key == "theta_posterior":
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(subspace.shape), 8),
                    nn.ReLU()
                )
                total_concat_size += 8

        self.extractors = nn.ModuleDict(extractors)

        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            if key in observations:
                encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)

class BeliefWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, decay: float=0.5, memory_decay_model: str='perfect'):
        super().__init__(env)
        self.env = self.unwrapped
        self.grid_size = self.env.grid_size
        self.num_tgt = 2
        self.decay = decay

        self.belief = Belief(self.env)
        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=float),
            "target": spaces.Box(low=0, high=1, shape=(1, self.num_tgt), dtype=float),
        })
        
        self.history = []
        self.mem_model = memory_model[memory_decay_model]

    def reset(self, seed=None, **kwargs):
        self.history = []
        if 'env' in kwargs:
            self.env = kwargs['env'].unwrapped
        else:
            self.env.reset(seed=seed)
        obs = self.env._get_obs()
        
        
        self.belief = Belief(self.env)
        new_obs = {
            "obs": obs,
            "action": None,
            "step_count": 0,
        }
        self.history.append(new_obs)
        return self._get_belief(), {}
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_obs = {
            "obs": obs,
            "action": action,
            "step_count": self.history[-1]["step_count"] + 1,
        }
        self.history.append(new_obs)
        self._update_belief()
        return self._get_belief(), reward, terminated, truncated, {}

    def _get_history(self):
        return self.history
    
    def _get_belief(self):
        return {
            "position": self.belief.pos,
            "target": self.belief.tgt,
        }

    def _get_state(self):
        return self.env._get_state()
    
    def _get_observation(self, s=None):
        return self.env._get_obs(s)
    
    def _update_belief(self, history=None, theta=None):
        if history is None:
            history = self.history
        if theta is None:
            theta = self.decay
        self.belief.update(memory_decay_model=self.mem_model, history=history, decay=theta)
    
    def _compute_belief(self, history=None):
        if history is None:
            history = self.history
        return self.belief.compute(history=history)


class PersistentBeliefWrapper(BeliefWrapper):
    """Model B (persistent forgetting), a faithful subclass of BeliefWrapper that changes ONLY
    the memory update. 

    Model A (BeliefWrapper): self.history stays RAW; every step the belief is recomputed from the
    full raw history with a FRESH decay, so a dropped target can reappear next step (resample-fresh).

    Model B (this class): the decay is PERSISTENT. Each step the stored memory itself is decayed, that is, 
    every still-remembered target is dropped with probability `decay` and STAYS dropped, so
    self.history is the accumulated-decay memory and P(a sighting survives n steps) = (1-decay)^n.
    The belief is then recomputed over that already-decayed memory with no further per-step drop.
    """

    def __init__(self, env: gym.Env, decay: float = 0.5, memory_decay_model: str = 'rand_drop'):
        super().__init__(env, decay=decay, memory_decay_model='perfect')
        self._drop = memory_model[memory_decay_model]

    def _update_belief(self, history=None, theta=None):
        if theta is None:
            theta = self.decay
        self.history = self._drop(self.history, theta)
        super()._update_belief(history=self.history, theta=theta)


