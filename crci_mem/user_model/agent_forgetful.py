from .agent_base import BaseAgent
from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
from crci_mem.user_model.wrappers import BeliefWrapper, PersistentBeliefWrapper
from stable_baselines3.common.preprocessing import preprocess_obs
import torch.nn.functional as F
import torch
from copy import deepcopy


def make_belief_wrapper(env, theta, memory_model='A'):
    """Wrap env with the user's memory model. 'A' = BeliefWrapper (resample-fresh, default),
    'B' = PersistentBeliefWrapper (persistent forgetting; used for the assistance experiment)."""
    if memory_model == 'B':
        return PersistentBeliefWrapper(env, decay=theta)
    return BeliefWrapper(env, decay=theta, memory_decay_model='rand_drop')


class ForgetfulHH(BaseAgent):
    def __init__(self, 
                 env: gym.Env=None,
                 model_cls=PPO,
                 theta: float=0, 
                 seed: int=42,
                 save_dir: str=None,
                 true_policy_dir=None,
                 device: str='cuda',
                 deterministic=True,
                 temperature=None,
                 entropy_threshold: float=0.8,  # Threshold for accepting action hints
                 memory_model: str='A'):        # 'A' resample-fresh (default) or 'B' persistent
        self.theta = theta
        self.memory_model = memory_model
        if env is not None:
            self.env = env.unwrapped
            self.wrapper = make_belief_wrapper(env, theta, memory_model)
        else:
            self.env = None
            self.wrapper = None
        self.save_dir = save_dir
        self.seed = seed
        self.device = device
        np.random.seed(seed)
        self.policy_dir = true_policy_dir
        self.model = model_cls
        if self.policy_dir is not None:
            self.policy = self.model.load(self.policy_dir, device=self.device)
        else:
            self.policy = None
        self.deterministic = deterministic
        self.temperature = temperature

        self.current_assistance = None
        self.assistance_type = None  # Can be None, 'action', or 'memory'
        self.entropy_threshold = entropy_threshold  
        self.reset()

    def reset(self, env: gym.Env = None, seed: int = None, load_policy: bool = False):
        if seed is None:
            seed = self.seed
        if env is None:
            env = self.env
        if env is not None:
            env = env.unwrapped
            self.env = env
            if self.wrapper is None:
                self.wrapper = make_belief_wrapper(env, self.theta, self.memory_model)
            reset_kwargs = {
                'env': env,
            }
            self.wrapper.reset(**reset_kwargs)
            self.s = self.wrapper._get_state()
            self.o = self.wrapper._get_observation()
            self.b = self.wrapper._get_belief()
        else:
            self.wrapper = None
            self.s = None
            self.o = None
            self.b = None
        self.history = []
        self.current_assistance = None
        self.assistance_type = None
        if load_policy:
            self.load_policy(self.policy_dir)

    def load_policy(self, policy_dir):
        self.policy = self.model.load(policy_dir, device=self.device)

    def adaptive_temperature(self, current_step, initial_temp=5.0):
        decay_rate = 0.01 * np.log(initial_temp)
        return initial_temp * np.exp(-decay_rate * current_step)
    
    def predict(self, b, temperature=3.0):
        """Get action from policy with temperature scaling."""
        policy = self.policy.policy
        policy.set_training_mode(False)
        with torch.no_grad():  
            obs_tensor, _ = policy.obs_to_tensor(b)
            preprocessed_obs = preprocess_obs(obs_tensor, policy.observation_space, normalize_images=policy.normalize_images)
            features = policy.pi_features_extractor(preprocessed_obs)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)
            scaled_logits = logits / temperature
            action_probs = F.softmax(scaled_logits, dim=-1)
            action = torch.multinomial(action_probs, num_samples=1)
            action = action.cpu().numpy().squeeze()
        return action, action_probs
    
    def compute_entropy(self, policy=None, b=None, temperature=None):
        """
        Compute the entropy of a probability distribution.
        Higher entropy = more uncertainty.
        
        Args:
            probs: Action probability distribution tensor
            
        Returns:
            entropy: Normalized entropy value [0,1]
        """
        if policy is None:
            policy = self.policy.policy
        if b is None:
            b = self.b
        if temperature is None:
            temperature = self.temperature  
        if temperature is None:
            policy.set_training_mode(False)
            obs_tensor, _ = policy.obs_to_tensor(b)
            dist = policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs
        else:
            _, probs = self.predict(b, temperature=temperature)
        eps = 1e-8
        probs_safe = torch.clamp(probs, min=eps, max=1.0)
        log_probs = torch.log(probs_safe)
        entropy = -torch.sum(probs_safe * log_probs, dim=-1)
        n_actions = probs.shape[-1]
        max_entropy = np.log(n_actions)
        normalized_entropy = entropy.item() / max_entropy
        
        return normalized_entropy
    
    def receive_assistance(self, assistance=None, assistance_type=None):
        """
        Receive assistance from the AI assistant.
        
        Args:
            assistance: The assistance content 
                - for action hints: the action to take
                - for memory hints: tuple of (time_step, observation)
            assistance_type: Type of assistance ('action' or 'memory')
        """
        self.current_assistance = assistance
        self.assistance_type = assistance_type

    def action_reaction_memory(self, a_content, history=None, theta=None):
        if history is None:
            history = self.wrapper.history
        if theta is None:
            theta = self.theta
        b = self.wrapper._get_belief()
        time_step, memory_hint = a_content
        if time_step < len(history):
            memory_hint_copy = deepcopy(memory_hint)
            original_entry = history[time_step]
            updated_entry = deepcopy(original_entry)
            updated_entry["obs"] = memory_hint_copy
            history[time_step] = updated_entry
            b = self.wrapper._compute_belief(history=history)
        return history, b

    def action_reaction(self, a_type=None, a_content=None):
        if a_type is None:
            a_type = self.assistance_type
        if a_content is None:
            a_content = self.current_assistance
        if a_type == 'action' and a_content is not None:
            entropy = self.compute_entropy()
            if entropy > self.entropy_threshold:
                return a_content
        elif a_type == 'memory' and a_content is not None:
            _, b = self.action_reaction_memory(a_content)
            self.b = b
        if self.temperature is None or self.temperature == 1.0:
            action, _ = self.policy.predict(self.b, deterministic=self.deterministic)
        else:
            action, _ = self.predict(self.b, temperature=self.temperature)
        return action
        

    def step(self):
        if self.assistance_type is not None and self.current_assistance is not None:
            action = self.action_reaction()
        else:
            if self.temperature is None or self.temperature == 1.0:
                action, _ = self.policy.predict(self.b, deterministic=self.deterministic)
            else:
                action, _ = self.predict(self.b, temperature=self.temperature)
        new_b, reward, terminated, truncated, info = self.wrapper.step(action)
        self.s = self.wrapper._get_state()
        self.o = self.wrapper._get_observation()
        self.b = new_b
        self.current_assistance = None
        self.assistance_type = None
        
        if terminated or truncated:
            done = True
        else:
            done = False
        return None, action, self.b, reward, done

    def sample_observation(self, s=None):
        return self.wrapper._get_observation(s)

    def update_belief(self, history=None, theta=None):
        if history is None:
            history = self.wrapper.history
        if theta is None:
            theta = self.theta
        self.wrapper.history = history
        self.wrapper.decay = theta
        self.wrapper._update_belief()
        return self.wrapper._get_belief()
    
    def _get_belief(self):
        return self.wrapper._get_belief()

    def compute_policy_probs(self, b, model):
        # to get the likelihood of action a: np.exp(policy.get_distribution(b).log_prob(a))
        device = self.device if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():  
            b_tensor = {
                "position": torch.tensor(b["position"], dtype=torch.float32).unsqueeze(0).to(device),
                "target": torch.tensor(b["target"], dtype=torch.float32).to(device)
            }
            return model.policy.get_distribution(b_tensor)


class ForgetfulHH_no_reaction(BaseAgent):
    def __init__(self, 
                 env: gym.Env,
                 model_cls=PPO,
                 theta: float=0, 
                 seed: int=42,
                 save_dir: str=None,
                 true_policy_dir=None,
                 device: str='cuda',
                 deterministic=True,
                 temperature=None,
                 memory_model: str='A'):
        self.theta = theta
        self.memory_model = memory_model
        self.env = env.unwrapped
        self.wrapper = make_belief_wrapper(env, theta, memory_model)
        self.save_dir = save_dir
        self.seed = seed
        self.device = device
        np.random.seed(seed)
        self.policy_dir = true_policy_dir
        self.model = model_cls
        if self.policy_dir is not None:
            self.policy = self.model.load(self.policy_dir, device=self.device)
        else:
            self.policy = None
        self.deterministic = deterministic
        self.temperature = temperature
        self.reset()

    def reset(self, env: gym.Env = None, seed: int = None, load_policy: bool = False):
        if seed is None:
            seed = self.seed
        if env is None:
            env = self.env
        self.env = env.unwrapped
        reset_kwargs = {
            'env': env,
        }
        self.wrapper.reset(**reset_kwargs)
        self.s = self.wrapper._get_state()
        self.o = self.wrapper._get_observation()
        self.b = self.wrapper._get_belief()
        self.history = []
        if load_policy:
            self.load_policy(self.policy_dir)

    def load_policy(self, policy_dir):
        self.policy = self.model.load(policy_dir, device=self.device)

    def predict(self, b, temperature=3.0):
        """Get action from policy with temperature scaling."""
        policy = self.policy.policy
        policy.set_training_mode(False)
        with torch.no_grad():  
            obs_tensor, _ = policy.obs_to_tensor(b)
            preprocessed_obs = preprocess_obs(obs_tensor, policy.observation_space, normalize_images=policy.normalize_images)
            features = policy.pi_features_extractor(preprocessed_obs)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)
            scaled_logits = logits / temperature
            action_probs = F.softmax(scaled_logits, dim=-1)
            action = torch.multinomial(action_probs, num_samples=1)
            action = action.cpu().numpy().squeeze()
        return action, action_probs
    
    def step(self):
        if self.temperature is None or self.temperature == 1.0:
            action, _ = self.policy.predict(self.b, deterministic=self.deterministic)
        else:
            action, _ = self.predict(self.b, temperature=self.temperature)
        new_b, reward, terminated, truncated, info = self.wrapper.step(action)
        self.s = self.wrapper._get_state()
        self.o = self.wrapper._get_observation()
        self.b = new_b
        if terminated or truncated:
            done = True
        else:
            done = False
        return None, action, self.b, reward, done

    def sample_observation(self, s=None):
        #self.wrapper.env.agent_pos = s
        return self.wrapper._get_observation(s)

    def update_belief(self, history=None, theta=None):
        if history is None:
            history = self.wrapper.history
        if theta is None:
            theta = self.theta
        self.wrapper.history = history
        self.wrapper.decay = theta
        self.wrapper._update_belief()
        return self.wrapper._get_belief()
    
    def _get_belief(self):
        return self.wrapper._get_belief()

    def compute_policy_probs(self, b, model):
        # to get the likelihood of action a: np.exp(policy.get_distribution(b).log_prob(a))
        device = self.device if torch.cuda.is_available() else 'cpu'
        b_tensor = {
            "position": torch.tensor(b["position"], dtype=torch.float32).unsqueeze(0).to(device),
            "target": torch.tensor(b["target"], dtype=torch.float32).to(device)
        }
        return model.policy.get_distribution(b_tensor)
    