from stable_baselines3 import PPO
from crci_mem.inference.base import BaseNestedParticleFilter
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from copy import deepcopy
from stable_baselines3.common.preprocessing import preprocess_obs

ASSIST_TYPES = {
        "NO_ASSIST": 0,      # No assistance
        "ACTION_HINT": 1,    # Provide action hint
        "MEMORY_HINT": 2     # Provide memory hint
    }

class NestedParticleFilter(BaseNestedParticleFilter):
    def __init__(self, candidate_thetas, M, simulated_agent_class, simulated_agent_params, theta2policy_dir, model_class, seed=42, device='cuda', temperature=1.0, adaptive=False):
        self.candidate_thetas = np.array(candidate_thetas)
        self.M = M
        self.theta_particles = {}
        for theta in self.candidate_thetas:
            self.theta_particles[theta] = {
                'inner_particles': [],
                'theta_weight': 1.0
            }
        self.theta_particles_history = {}
        self.agent = simulated_agent_class(**simulated_agent_params)
        self.time_step = 0
        self.seed = seed
        self.theta2policy_dir = theta2policy_dir
        self.device = device
        self.temperature = temperature
        self.adaptive = adaptive
        self.model_class = model_class
        # Use policy caching for improved performance
        self._policy_cache = {}
        self.policies = self.load_policies()
        self.eps = 1e-10  

    def reset(self, env: gym.Env, seed: int = None):
        """
        Full reset of the CRCI model, including both theta posterior and belief tracking.
        Use this when switching to a completely new agent.
        """
        if seed is None:
            seed = self.seed
        self.agent.reset(env=env, seed=seed, load_policy=False)
        self.theta_particles = {}
        for theta in self.candidate_thetas:
            self.theta_particles[theta] = {
                'inner_particles': [],
                'theta_weight': 1.0
            }
        self.theta_particles_history = {}
        self.time_step = 0

    def reset_streaming(self, env: gym.Env, seed: int = None):
        if seed is None:
            seed = self.seed
        self.agent.reset(env=env, seed=seed, load_policy=False)
        
    
    def load_policies(self):
        theta2policy = {}
        for theta in self.candidate_thetas:
            policy_dir = self.theta2policy_dir[theta]
            if policy_dir in self._policy_cache:
                theta2policy[theta] = self._policy_cache[policy_dir]
            else:
                try:
                    model = self.model_class.load(policy_dir, device=self.device)
                    self._policy_cache[policy_dir] = model
                    theta2policy[theta] = model
                except Exception as e:
                    print(f"Error loading policy for theta={theta}: {e}")
        return theta2policy

    def initialize(self, s0, a0, streaming=False):
        for idx, theta in enumerate(self.candidate_thetas):
            inner_particles = []
            likelihoods = []
            o0 = self.sample_observation(s0)
            m0 = [{
                'obs': o0,
                'action': None,
                'action_new': a0,
                'step_count': 0,
            }]
            b0 = self.agent._get_belief()
            for _ in range(self.M):
                lkh = self.compute_likelihood(b0, a0, self.policies[theta], self.temperature, self.adaptive)
                lkh = max(lkh, self.eps)
                inner_particles.append({'m': deepcopy(m0), 'b': b0, 'weight': lkh})
                likelihoods.append(lkh)
            likelihoods = np.array(likelihoods)
            sum_likelihoods = np.sum(likelihoods)
            norm_weights = likelihoods / sum_likelihoods if sum_likelihoods > self.eps else np.ones(self.M)/self.M
            for j in range(self.M):
                inner_particles[j]['weight'] = norm_weights[j]
            self.theta_particles[theta]['inner_particles'] = inner_particles
            avg_L = np.mean(likelihoods)
            if streaming:
                self.theta_particles[theta]['theta_weight'] = max(self.theta_particles[theta]['theta_weight'] * avg_L, self.eps)
            else:
                self.theta_particles[theta]['theta_weight'] = max(avg_L, self.eps)
        all_weights = np.array([self.theta_particles[theta]['theta_weight'] for theta in self.candidate_thetas])
        sum_all_weights = np.sum(all_weights)
        norm_all = all_weights / sum_all_weights if sum_all_weights > self.eps else np.ones(len(self.candidate_thetas))/len(self.candidate_thetas)
        for idx, theta in enumerate(self.candidate_thetas):
            self.theta_particles[theta]['theta_weight'] = norm_all[idx]
        self.theta_particles_history[self.time_step] = {}
        self.theta_particles_history[self.time_step]['posterior'] = self.get_posterior()
        self.theta_particles_history[self.time_step]['inner_posteriors'] = {theta: self.get_inner_posterior(theta) for theta in self.candidate_thetas}
        self.time_step += 1

    def update(self, s, a):
        for theta in self.candidate_thetas:
            inner_particles = self.theta_particles[theta]['inner_particles']
            likelihoods = []
            o_new = self.sample_observation(s)
            for idx, particle in enumerate(inner_particles):
                m_new = self.update_memory(particle['m'], o_new, a)
                particle['m'].append(m_new)
                b_new = self.update_belief(particle['m'], theta)
                lkh = self.compute_likelihood(b_new, a, self.policies[theta], self.temperature, self.adaptive)
                lkh = max(lkh, self.eps)
                particle['b'] = b_new
                particle['weight'] = max(particle['weight'] * lkh, self.eps)
                likelihoods.append(lkh)
            weights = np.array([p['weight'] for p in inner_particles])
            sum_weights = np.sum(weights)
            norm_weights = weights / sum_weights if sum_weights > self.eps else np.ones(self.M)/self.M
            for idx, particle in enumerate(inner_particles):
                particle['weight'] = norm_weights[idx]
            avg_L = np.mean(likelihoods)
            self.theta_particles[theta]['theta_weight'] = max(self.theta_particles[theta]['theta_weight'] * sum_weights, self.eps)
        candidate_weights = np.array([self.theta_particles[theta]['theta_weight'] for theta in self.candidate_thetas])
        sum_candidate_weights = np.sum(candidate_weights)
        candidate_weights = candidate_weights / sum_candidate_weights if sum_candidate_weights > self.eps else np.ones(len(self.candidate_thetas))/len(self.candidate_thetas)
        for idx, theta in enumerate(self.candidate_thetas):
            self.theta_particles[theta]['theta_weight'] = candidate_weights[idx]
        self.theta_particles_history[self.time_step] = {}
        self.theta_particles_history[self.time_step]['posterior'] = self.get_posterior()
        self.theta_particles_history[self.time_step]['inner_posteriors'] = {theta: self.get_inner_posterior(theta) for theta in self.candidate_thetas}
        self.time_step += 1  

    def get_posterior(self):
        return {theta: self.theta_particles[theta]['theta_weight'] for theta in self.candidate_thetas}
    
    def get_inner_posterior(self, theta):
        return {idx: particle for idx, particle in enumerate(self.theta_particles[theta]['inner_particles'])}
    
    def get_particle_history(self):
        return self.theta_particles_history
    
    def sample_observation(self, s):
        return self.agent.sample_observation(s)

    def update_memory(self, m, o_new, a):
        new_m = {
            'obs': o_new,
            'action': m[-1]['action_new'],
            'action_new': a,
            'step_count': m[-1]['step_count'] + 1,
        }
        return new_m

    def update_belief(self, m, theta):
        return self.agent.update_belief(m, theta)
    
    def adaptive_temperature(self, current_step, initial_temp=5.0):
        """
        Compute the adaptive temperature based on the current step.
        """
        decay_rate = 0.01 * np.log(max(initial_temp, 1.0))  
        return initial_temp * np.exp(-decay_rate * current_step)
        
    def compute_likelihood(self, b, a, model, temperature=1.0, adaptive=False):
        """Likelihood p(a | b ; theta) of action a under belief b."""
        with torch.no_grad():
            policy = model.policy
            device = self.device if torch.cuda.is_available() else 'cpu'
            
            b_tensor = {
                "position": torch.tensor(b["position"], dtype=torch.float32).unsqueeze(0).to(device),
                "target": torch.tensor(b["target"], dtype=torch.float32).to(device)
            }
            
            preprocessed_obs = preprocess_obs(b_tensor, policy.observation_space, normalize_images=policy.normalize_images)
            features = policy.pi_features_extractor(preprocessed_obs)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)
            
            if adaptive:
                temperature = max(self.adaptive_temperature(current_step=self.time_step), self.eps)
            else:
                temperature = max(temperature, self.eps)
                
            scaled_logits = logits / temperature
            action_probs = F.softmax(scaled_logits, dim=-1)
            likelihood = action_probs[0][a].item()
            likelihood = max(likelihood, self.eps)
            
        return likelihood

class AssistantParticleFilter(BaseNestedParticleFilter):
    def __init__(self, candidate_thetas, M, simulated_agent_class, simulated_agent_params, theta2policy_dir, model_class, seed=42, device='cuda', temperature=1.0, adaptive=False):
        self.candidate_thetas = np.array(candidate_thetas)
        self.M = M
        self.theta_particles = {}
        for theta in self.candidate_thetas:
            self.theta_particles[theta] = {
                'inner_particles': [],
                'theta_weight': 1.0
            }
        self.theta_particles_history = {}
        self.agent_template = simulated_agent_class(**simulated_agent_params)  
        self.time_step = 0
        self.seed = seed
        self.theta2policy_dir = theta2policy_dir
        self.device = device
        self.temperature = temperature
        self.adaptive = adaptive
        self.model_class = model_class
        self._policy_cache = {}
        self.policies = self.load_policies()

    def reset(self, env: gym.Env, seed: int = None):
        """
        Full reset of the CRCI model, including both theta posterior and belief tracking.
        Use this when switching to a completely new agent.
        """
        if seed is None:
            seed = self.seed
        self.agent_template.reset(env=env, seed=seed, load_policy=False)
        self.theta_particles = {}
        for theta in self.candidate_thetas:
            self.theta_particles[theta] = {
                'inner_particles': [],
                'theta_weight': 1.0
            }
        self.theta_particles_history = {}
        self.time_step = 0

    def reset_streaming(self, env: gym.Env, seed: int = None):
        if seed is None:
            seed = self.seed
        self.agent_template.reset(env=env, seed=seed, load_policy=False)
    
    def load_policies(self):
        theta2policy = {}
        for theta in self.candidate_thetas:
            policy_dir = self.theta2policy_dir[theta]
            if policy_dir in self._policy_cache:
                theta2policy[theta] = self._policy_cache[policy_dir]
            else:
                model = self.model_class.load(policy_dir, device=self.device)
                self._policy_cache[policy_dir] = model
                theta2policy[theta] = model
        return theta2policy
    
    def _calculate_temperature(self):
        """Calculate temperature parameter for action distribution scaling"""
        if self.adaptive:
            if hasattr(self.agent_template, 'adaptive_temperature'):
                return self.agent_template.adaptive_temperature(current_step=self.time_step, initial_temp=self.temperature)
            decay_rate = 0.01 * np.log(self.temperature) if self.temperature > 0 else 0.01
            return self.temperature * np.exp(-decay_rate * self.time_step)
        return self.temperature

    def _batch_compute_likelihoods(self, beliefs_batch, action, policy_model, a_t_ai=None, a_t_ai_content=None):
        """
        Compute action likelihoods in batch mode for improved efficiency.
        
        Args:
            beliefs_batch: List of belief states
            action: The action to compute likelihood for
            policy_model: The policy model for a specific theta
            a_t_ai: Type of AI assistance provided for action a_t (e.g., 'suggestion', 'correction').
            a_t_ai_content: Specific content of the AI assistance.
        Returns:
            np.ndarray of likelihoods for each belief
        """
        if not beliefs_batch:
            return np.array([])
    
        temperature = self._calculate_temperature()
        
        try:
            with torch.no_grad():  
                positions = torch.stack([
                    torch.tensor(b['position'], dtype=torch.float32) 
                    for b in beliefs_batch
                ]).to(self.device)
                
                targets = torch.stack([
                    torch.tensor(b['target'], dtype=torch.float32) 
                    for b in beliefs_batch
                ]).to(self.device)
                
                batched_obs = {
                    "position": positions,
                    "target": targets
                }
                
                policy = policy_model.policy
                policy.set_training_mode(False)
                
                preprocessed_obs = preprocess_obs(batched_obs, policy.observation_space, 
                                                normalize_images=policy.normalize_images)
                features = policy.pi_features_extractor(preprocessed_obs)
                latent_pi = policy.mlp_extractor.forward_actor(features)
                logits = policy.action_net(latent_pi)
                
                scaled_logits = logits / temperature
                action_probs = F.softmax(scaled_logits, dim=-1)
                
                action_likelihoods = action_probs[:, action].cpu().numpy()
                if a_t_ai is not None and a_t_ai == ASSIST_TYPES["ACTION_HINT"] and a_t_ai_content is not None:
                    eps = 1e-8
                    probs_safe = torch.clamp(action_probs, min=eps, max=1.0) # action_probs is a tensor
                    log_probs = torch.log(probs_safe)
                    entropy_tensor = -torch.sum(probs_safe * log_probs, dim=-1) # This is a batch of entropies (Tensor)
                
                    n_actions = action_probs.shape[-1]
                    if n_actions > 1:
                        max_entropy = np.log(n_actions)
                        if max_entropy > 0:
                            normalized_entropy_tensor = entropy_tensor / max_entropy
                            normalized_entropy_np = normalized_entropy_tensor.cpu().numpy()
                        else: 
                            normalized_entropy_np = torch.zeros_like(entropy_tensor).cpu().numpy()
                    else: 
                        normalized_entropy_np = torch.zeros_like(entropy_tensor).cpu().numpy()

                    # adjust likelihood by entropy: if entropy > threshold the user takes the AI action hint,
                    # so likelihood is 1 for action==hint and 0 otherwise; else the policy action_likelihoods (batched)
                    high_entropy_mask = normalized_entropy_np > self.agent_template.entropy_threshold
                    value_if_high_entropy = 1.0 if action == a_t_ai_content else 0.0
                    action_likelihoods = np.where(high_entropy_mask, value_if_high_entropy, action_likelihoods)
                
            return action_likelihoods
            
        except Exception as e:
            print(f"Warning: Batch likelihood calculation failed: {e}. Falling back to individual processing.")
            likelihoods = []
            
            for belief in beliefs_batch:
                with torch.no_grad():
                    likelihood = self.compute_likelihood(belief, action, policy_model, temperature)
                likelihoods.append(likelihood)
                
            return np.array(likelihoods)

    def initialize(self, s0, a0, streaming=False):
        """Initialize particles based on user's logic: b0 is uniform"""
        step_history_log = {}
        o0 = self.sample_observation(s0)
        m0_segment = {
            'obs': o0,
            'action': None, # Action leading to s0 unknown
            'action_new': a0, # Action taken from s0
            'step_count': 0,
        }
        for theta in self.candidate_thetas:
            # uniform initial belief
            b0 = self.agent_template._get_belief()
            beliefs_batch = [b0] * self.M
            # Likelihood P(a0 | b0, policy)
            likelihoods = self._batch_compute_likelihoods(
                beliefs_batch, 
                a0, 
                self.policies[theta]
            )
            inner_particles = []
            for j in range(self.M):
                inner_particles.append({
                    'm': [m0_segment.copy()], 
                    'b': b0.copy(),                  
                    'weight': likelihoods[j]
                })
            weights = np.array([p['weight'] for p in inner_particles])
            sum_weights = np.sum(weights)
            norm_weights = weights / sum_weights if sum_weights > 0 else np.ones(self.M) / self.M
            for j in range(self.M):
                inner_particles[j]['weight'] = norm_weights[j]
            self.theta_particles[theta]['inner_particles'] = inner_particles
            avg_L = np.mean(likelihoods)
            if streaming and self.time_step > 0:
                current_theta_weight = self.theta_particles[theta]['theta_weight']
                self.theta_particles[theta]['theta_weight'] = (
                    current_theta_weight * avg_L if current_theta_weight > 1e-10 else 1e-10 * avg_L
                )
            else:
                self.theta_particles[theta]['theta_weight'] = avg_L
        all_weights = np.array([self.theta_particles[theta]['theta_weight'] for theta in self.candidate_thetas])
        sum_all_weights = np.sum(all_weights)
        norm_all = all_weights / sum_all_weights if sum_all_weights > 0 else np.ones(len(self.candidate_thetas)) / len(self.candidate_thetas)
        for idx, theta in enumerate(self.candidate_thetas):
            self.theta_particles[theta]['theta_weight'] = norm_all[idx]
        step_history_log['posterior'] = self.get_posterior()
        step_history_log['inner_posteriors'] = {
            theta: self.get_inner_posterior(theta) for theta in self.candidate_thetas
        }
        self.theta_particles_history[self.time_step] = step_history_log
        self.time_step += 1

    def update(self, s_t, a_t, a_t_ai=None, a_t_ai_content=None):
        """
        Update particle filter based on state s_t and action a_t taken from it.
        User logic: 
        1. Get o_t. 
        2. Update memory to m_t. 
        3. Compute b_t.
        4. Compute P(a_t|b_t). 
        5. Update weights.
        
        Args:
            s_t: Current state.
            a_t: Action taken by the user (potentially influenced by AI).
            a_t_ai: Type of AI assistance provided for action a_t (e.g., 'suggestion', 'correction').
            a_t_ai_content: Specific content of the AI assistance.
        """
        o_t = self.sample_observation(s_t.copy()) # Observation at time t
        
        for theta in self.candidate_thetas:
            inner_particles = self.theta_particles[theta]['inner_particles']
            
            beliefs_t_batch = [] 
            updated_particles_data = [] 
            
            for particle in inner_particles:
                # particle['m'] currently holds history up to m_{t-1}
                m_t_segment = self.update_memory(particle['m'], o_t.copy(), a_t)
                
                # Construct the full memory history up to time t
                m_t = particle['m'] + [m_t_segment]

                if a_t_ai is not None and a_t_ai == ASSIST_TYPES["MEMORY_HINT"] and a_t_ai_content is not None:
                    m_t, b_t = self.agent_template.action_reaction_memory(a_t_ai_content, m_t, theta)
                else:
                    b_t = self.update_belief(m_t, theta)
                
                beliefs_t_batch.append(b_t.copy())
                updated_particles_data.append({'m': m_t.copy(), 'b': b_t.copy()})

            likelihoods = self._batch_compute_likelihoods(
                beliefs_t_batch, 
                a_t,             
                self.policies[theta],
                a_t_ai,
                a_t_ai_content,
            )
            
            # Update particle weights and store final states
            for idx, particle in enumerate(inner_particles):
                # Update weight: w_t = w_{t-1} * P(a_t | b_t ; theta)
                particle['weight'] *= likelihoods[idx]
                particle['m'] = updated_particles_data[idx]['m']
                particle['b'] = updated_particles_data[idx]['b']

            # Normalize inner particle weights 
            weights = np.array([p['weight'] for p in inner_particles])
            sum_weights = np.sum(weights)
            norm_weights = weights / sum_weights if sum_weights > 0 else np.ones(self.M) / self.M
            for idx, particle in enumerate(inner_particles):
                particle['weight'] = norm_weights[idx]
            
            # Update theta weights 
            # sum_weights = \sum_i w_{t-1,i}*p(a|b_i ; theta)
            current_theta_weight = self.theta_particles[theta]['theta_weight']
            self.theta_particles[theta]['theta_weight'] = (
                current_theta_weight * sum_weights if current_theta_weight > 1e-10 else 1e-10 * sum_weights
            )
        
        # Normalize theta weights 
        candidate_weights = np.array([
            self.theta_particles[theta]['theta_weight'] for theta in self.candidate_thetas
        ])
        sum_candidate_weights = np.sum(candidate_weights)
        norm_candidate_weights = candidate_weights / sum_candidate_weights if sum_candidate_weights > 0 else np.ones(len(self.candidate_thetas)) / len(self.candidate_thetas)
        for idx, theta in enumerate(self.candidate_thetas):
            self.theta_particles[theta]['theta_weight'] = norm_candidate_weights[idx]
        
        # Store history 
        self.theta_particles_history[self.time_step] = {
            'posterior': self.get_posterior(),
            'inner_posteriors': {theta: self.get_inner_posterior(theta) for theta in self.candidate_thetas}
        }
        self.time_step += 1
        
        
    def compute_likelihood(self, b, a, model, temperature=None):
        """Compute likelihood p(a | b ; theta) with the given policy model"""
        if temperature is None:
            temperature = self._calculate_temperature()
        
        with torch.no_grad():
            policy = model.policy
            device = self.device if torch.cuda.is_available() else 'cpu'
            
            b_tensor = {
                "position": torch.tensor(b["position"], dtype=torch.float32).unsqueeze(0).to(device),
                "target": torch.tensor(b["target"], dtype=torch.float32).to(device)
            }
            
            preprocessed_obs = preprocess_obs(b_tensor, policy.observation_space, normalize_images=policy.normalize_images)
            features = policy.pi_features_extractor(preprocessed_obs)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)
            
            scaled_logits = logits / temperature
            action_probs = F.softmax(scaled_logits, dim=-1)
            
            likelihood = action_probs[0][a].item()
        
        return likelihood

    def get_posterior(self):
        """Get posterior over theta values"""
        return {theta: self.theta_particles[theta]['theta_weight'] for theta in self.candidate_thetas}
    
    def get_inner_posterior(self, theta):
        """Get posterior over inner particles for a specific theta"""
        return {idx: particle for idx, particle in enumerate(self.theta_particles[theta]['inner_particles'])}
    
    def get_particle_history(self):
        """Get particle history for visualization/analysis"""
        return self.theta_particles_history
    
    def sample_observation(self, s):
        """Sample observation from environment state"""
        return self.agent_template.sample_observation(s)

    def update_memory(self, m, o_t, a_t):
        """
        Create a new memory segment for time t based on user's logic.
        m: memory history up to t-1
        o_t: observation received at state s_t
        a_t: action taken from state s_t
        """
        new_m_segment = {
            'obs': o_t,
            # Action that led to o_t was the 'action_new' from the previous step (a_{t-1})
            'action': m[-1]['action_new'] if m and 'action_new' in m[-1] else None, 
            # Action taken from o_t is a_t
            'action_new': a_t,                
            'step_count': m[-1]['step_count'] + 1 if m else 0,
        }
        return new_m_segment

    def update_belief(self, m, theta):
        """Update belief given memory history m (up to time t) and theta"""
        return self.agent_template.update_belief(m, theta)
    
    def adaptive_temperature(self, current_step, initial_temp=5.0):
        """
        Compute the adaptive temperature based on the current step.
        This is a legacy method, use _calculate_temperature instead.
        """
        decay_rate = 0.01 * np.log(initial_temp)
        return initial_temp * np.exp(-decay_rate * current_step)
