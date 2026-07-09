import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf
import torch
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
import os
from crci_mem.inference.nested_particle_filter import AssistantParticleFilter
from crci_mem.user_model.agent_forgetful import ForgetfulHH
from stable_baselines3 import PPO
import time



class AdaptiveAssistanceEnv(gym.Env):
    """
    Environment wrapper for training an adaptive AI assistant via PPO.
    This environment handles the interaction between the AI assistant and a forgetful agent.
    The AI assistant observes the agent's inferred belief state and decides on the type of assistance.
    """
    ASSIST_TYPES = {
        "NO_ASSIST": 0,      # No assistance
        "ACTION_HINT": 1,    # Provide action hint
        "MEMORY_HINT": 2     # Provide memory hint
    }
    
    def __init__(
        self,
        base_env: gym.Env,
        theta2policy_dir: Dict[float, str],
        crci_params: Dict,
        assistance_cost: Dict[int, float] = None,
        model_class=PPO,
        device: str = "cuda",
        seed: int = 42,
        optimal_policy_path: str = None,
        episodes_per_agent: int = 3,  # Number of episodes to run with same agent before switching
        entropy_threshold: float = 0.8,  # Threshold for agent's action hint acceptance
        temperature: float = 1.0,
        deterministic: bool = False,
        max_steps: int = 30,
        memory_model: str = 'B',   # user's memory model for assistance (persistent forgetting)
    ):
        self.base_env_wrapped = base_env
        self.memory_model = memory_model
        self.base_env = base_env.unwrapped
        # reset base_env with seed
        self.base_env.reset(seed=seed)
        self.theta2policy_dir = theta2policy_dir
        self.candidate_thetas = list(theta2policy_dir.keys())
        self.model_class = model_class
        self.device = device
        self.seed = seed
        self.optimal_policy_path = optimal_policy_path
        self.episodes_per_agent = episodes_per_agent
        self.entropy_threshold = entropy_threshold
        
        self.temperature = temperature
        self.deterministic = deterministic
        
        # Memory optimization: limit buffer size
        self.critical_obs_buffer_size = 3
        
        # Set default assistance costs if not provided
        if assistance_cost is None:
            self.assistance_cost = {
                self.ASSIST_TYPES["NO_ASSIST"]: 0.0,
                self.ASSIST_TYPES["ACTION_HINT"]: 0.1,  # Cost for providing action hint
                self.ASSIST_TYPES["MEMORY_HINT"]: 0.05  # Cost for providing memory hint
            }
        else:
            self.assistance_cost = assistance_cost

        
        # Load policies with caching
        self._policy_cache = {}
        self.theta_policies = self._load_theta_policies()
        self.optimal_policy = self._load_optimal_policy()
        
        # Setup CRCI (NPF) for belief inference
        self.crci_params = crci_params
        self.crci = AssistantParticleFilter(**crci_params)
        
        # Define observation and action spaces
        grid_size = self.base_env.grid_size
        num_tgt = self.base_env.num_tgt
        
        self.observation_space = gym.spaces.Dict({
            "position": gym.spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32),
            "target": gym.spaces.Box(low=0, high=1, shape=(1, num_tgt), dtype=np.float32),
            "theta_posterior": gym.spaces.Box(low=0, high=1, shape=(len(self.candidate_thetas),), dtype=np.float32)
        })
        
        self.action_space = gym.spaces.Discrete(len(self.ASSIST_TYPES))
        
        self.agent = None
        self.theta_true = None
        self.current_theta = None  # To track if we're continuing with the same agent or switching
        self.current_episode = 0  # To track which episode we're on with the current agent
        self.critical_observations = []
        self.step_count = 0
        #self.max_steps = self.base_env.max_steps
        self.max_steps = max_steps
        self.last_agent_action = None
        self.last_state = None
        self.last_agent_belief = None
        self.cumulative_reward = 0
        self.ai_belief = None
        
    def _load_theta_policies(self):
        """Load the agent policies for each theta with caching."""
        policies = {}
        for theta in self.candidate_thetas:
            policy_dir = self.theta2policy_dir[theta]
            if policy_dir in self._policy_cache:
                policies[theta] = self._policy_cache[policy_dir]
            else:
                policy = self.model_class.load(policy_dir, device=self.device)
                self._policy_cache[policy_dir] = policy
                policies[theta] = policy
        return policies
    
    def _load_optimal_policy(self):
        """Load the optimal MDP policy with caching."""
        if self.optimal_policy_path and os.path.exists(self.optimal_policy_path):
            if self.optimal_policy_path in self._policy_cache:
                return self._policy_cache[self.optimal_policy_path]
            policy = self.model_class.load(self.optimal_policy_path, device=self.device)
            self._policy_cache[self.optimal_policy_path] = policy
            return policy
        
        # Try to look for the optimal policy in the same directory as the agent policies
        sample_policy_path = next(iter(self.theta2policy_dir.values()))
        policy_dir = os.path.dirname(sample_policy_path)
        optimal_policy_path = os.path.join(policy_dir, 'ppo_mdp_best.zip')
        
        if os.path.exists(optimal_policy_path):
            if optimal_policy_path in self._policy_cache:
                return self._policy_cache[optimal_policy_path]
            policy = self.model_class.load(optimal_policy_path, device=self.device)
            self._policy_cache[optimal_policy_path] = policy
            return policy
        
        # Use the first theta policy as a fallback
        print("Warning: Optimal MDP policy not found. Using the policy for theta=0.0 as a fallback.")
        if 0.0 in self.theta_policies:
            return self.theta_policies[0.0]
        else:
            first_theta = min(self.candidate_thetas)
            return self.theta_policies[first_theta]
    
    def reset(self, theta: float = None, seed: Optional[int] = None, new_agent: bool = None) -> Tuple[Dict, Dict]:
        """Reset the environment with a specific or random theta value."""
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
        
        # Reset base environment
        obs, _ = self.base_env.reset()
        
        # Determine if we should start with a new agent or continue with the current one
        if new_agent is None:
            if self.current_episode >= self.episodes_per_agent or self.current_theta is None:
                new_agent = True
                self.current_episode = 0
            else:
                new_agent = False
                self.current_episode += 1
        
        # Select theta (randomly if not specified)
        if theta is None and new_agent:
            self.theta_true = np.random.choice(self.candidate_thetas)
        elif theta is not None and (new_agent or self.current_theta != theta):
            self.theta_true = theta
            new_agent = True  # Force new agent if theta changed
            self.current_episode = 0
        
        self.current_theta = self.theta_true
        
        # Initialize or reinitialize agent if needed
        if new_agent or self.agent is None:
            agent_params = {
                'env': self.base_env_wrapped,
                'model_cls': self.model_class,
                'theta': self.theta_true,
                'save_dir': '',
                'true_policy_dir': self.theta2policy_dir[self.theta_true],
                'seed': self.seed,
                'device': self.device,
                'entropy_threshold': self.entropy_threshold,
                'temperature': self.temperature,
                'deterministic': self.deterministic,
                'memory_model': self.memory_model,
            }
            self.agent = ForgetfulHH(**agent_params)
        else:
            # Just reset the existing agent for a new episode
            self.agent.reset(env=self.base_env_wrapped, seed=self.seed)
        
        # Reset CRCI appropriately
        if new_agent:
            # Full reset for new agent
            self.crci.reset(env=self.base_env, seed=self.seed)
        else:
            # Only reset belief tracking for new episode with same agent
            self.crci.reset_streaming(env=self.base_env, seed=self.seed)
        
        self.critical_observations = []
        self.step_count = 0
        self.last_agent_action = None
        self.last_state = None
        self.last_agent_belief = None
        self.cumulative_reward = 0
        
        
        s0 = self.base_env._get_state()
        b0 = self.agent.b
        self.last_state = s0  
        self.last_agent_belief = b0
        _, a0, b1, _, _ = self.agent.step()  
        self.last_agent_action = a0
        self.last_ai_action_type = self.ASSIST_TYPES["NO_ASSIST"]
        self.last_ai_action_content = None
        self.current_agent_belief = b1
        if new_agent:
            self.crci.initialize(s0, a0, streaming=False)
        else:
            self.crci.initialize(s0, a0, streaming=True)
        
        self.ai_belief = self._get_ai_belief()
        
        return self.ai_belief, {
            "theta_true": self.theta_true,
            "new_agent": new_agent,
            "episode": self.current_episode,
        }
    
    def _get_ai_belief(self) -> Dict:
        """Get the AI assistant's belief state efficiently."""
        # Get theta posterior
        theta_posterior = self.crci.get_posterior()
        theta_posterior_array = np.zeros(len(self.candidate_thetas), dtype=np.float32)
        
        for i, theta in enumerate(self.candidate_thetas):
            theta_posterior_array[i] = theta_posterior.get(theta, 0.0)
        
        # Initialize weighted belief state
        weighted_belief = {
            "position": np.zeros((self.base_env.grid_size, self.base_env.grid_size), dtype=np.float32),
            "target": np.zeros((1, self.base_env.num_tgt), dtype=np.float32)
        }
        
        # Compute weighted average over all thetas
        for theta in self.candidate_thetas:
            theta_weight = theta_posterior[theta]
            inner_posteriors = self.crci.theta_particles[theta]['inner_particles']
            
            if inner_posteriors:
                # Add weighted contribution from this theta's particles
                for i, particle in enumerate(inner_posteriors):
                    weighted_belief["position"] += theta_weight * particle['weight'] * particle['b']["position"]
                    weighted_belief["target"] += theta_weight * particle['weight'] * particle['b']["target"]
        
        # Check if we have any valid beliefs
        if np.sum(weighted_belief["position"]) > 0 and np.sum(weighted_belief["target"]) > 0:
            belief_state = weighted_belief
        else:
            # Fallback to uniform belief if no valid beliefs
            belief_state = {
                "position": np.ones((self.base_env.grid_size, self.base_env.grid_size), dtype=np.float32) / 
                           (self.base_env.grid_size * self.base_env.grid_size),
                "target": np.ones((1, self.base_env.num_tgt), dtype=np.float32) / self.base_env.num_tgt
            }
        
        return {
            "position": belief_state["position"].astype(np.float32),
            "target": belief_state["target"].astype(np.float32),
            "theta_posterior": theta_posterior_array
        }
    
    def _is_critical_observation(self, obs) -> bool:
        """Check if an observation is critical based on the target value."""
        return obs["target"] > 0
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Take a step with the AI assistant's action."""
        # action: a^ai_t
        self.step_count += 1
        
        action = int(action)
        
        # Get current state
        s_t = self.base_env._get_state()
        o_t = self.base_env._get_obs()
        
        assistance_type = None
        assistance_content = None
        hint_accepted = False
        entropy_value = None
        
        if action == self.ASSIST_TYPES["ACTION_HINT"]:
            # Provide action hint using the optimal MDP policy
            assistance_type = 'action'
            # Get the optimal action for the current state using the optimal MDP policy
            with torch.no_grad():  
                optimal_action, _ = self.optimal_policy.predict(s_t, deterministic=True)
            assistance_content = optimal_action
            
        elif action == self.ASSIST_TYPES["MEMORY_HINT"]:
            # Provide memory hint
            assistance_type = 'memory'
            # Randomly sample from critical observations
            if len(self.critical_observations) > 0:
                assistance_content = self.critical_observations[np.random.randint(0, len(self.critical_observations))]
            else:
                assistance_content = None

        # Check if current observation is critical and manage buffer size
        if self._is_critical_observation(o_t):
            self.critical_observations.append((self.step_count, deepcopy(o_t)))
            if len(self.critical_observations) > self.critical_obs_buffer_size:
                self.critical_observations.pop(0)
        
        # Provide assistance to the agent if applicable
        if assistance_type is not None and assistance_content is not None:
            self.agent.receive_assistance(assistance=assistance_content, assistance_type=assistance_type)
        
        # Store the current state before the agent takes action
        self.last_state = s_t
        
        # Agent takes a step with or without assistance
        _, agent_action, agent_belief_next, reward, done = self.agent.step()
        self.last_agent_action = agent_action
        self.last_agent_belief = self.current_agent_belief
        self.current_agent_belief = agent_belief_next
        
        # Calculate AI assistant's reward.
        ai_reward = reward - self.assistance_cost[action]
        self.cumulative_reward += ai_reward
        
        terminated = done
        truncated = self.step_count >= self.max_steps
        
        # Get AI assistant's next belief state
        self.crci.update(self.last_state, self.last_agent_action, action, assistance_content)
        self.last_ai_action_type = action
        self.last_ai_action_content = assistance_content
        self.ai_belief = self._get_ai_belief()
        
        info = {
            "cumulative_reward": self.cumulative_reward,
            "theta_true": self.theta_true,
            "assistance_type": action,
            "agent_action": agent_action,
            "agent_belief": self.last_agent_belief,
            "ai_belief": self.ai_belief,
            "current_episode": self.current_episode,
        }
        
        if entropy_value is not None:
            info["entropy"] = entropy_value
            info["hint_accepted"] = hint_accepted
        
        if terminated or truncated:
            ep_info = {
                'r': self.cumulative_reward,  # Total reward for the episode
                'l': self.step_count,         # Length of the episode
                't': time.time()              # Current time
            }
            info['episode'] = ep_info
        
        return self.ai_belief, ai_reward, terminated, truncated, info 

