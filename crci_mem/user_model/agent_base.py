from abc import ABC, abstractmethod
import numpy as np

class BaseAgent(ABC):
    def __init__(self, seed=42):
        self.seed = seed
        self.reset()
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def sample_observation(self, s):
        pass
    
    @abstractmethod
    def update_belief(self, m, o, theta):
        pass

    @abstractmethod
    def compute_policy_probs(self, b, target, beta):
        pass