from abc import ABC, abstractmethod
import numpy as np

class BaseNestedParticleFilter(ABC):
    """
    - candidate_thetas (or prior): list of candidate theta values
    - M: number of inner particles
    - memory model
    - observation model
    - belief update model
    - likelihood (action probability given belief from policy)
    Currently only consider discrete candidate thetas.
    """
    def __init__(self, candidate_thetas, M, agent, agent_params):
        self.candidate_thetas = np.array(candidate_thetas)
        self.M = M
        self.theta_particles = {}
        for theta in self.candidate_thetas:
            self.theta_particles[theta] = {
                'inner_particles': [],
                'theta_weight': 1.0
            }
        self.theta_particles_history = {}
        self.agent = agent(**agent_params)
        self.time_step = 0
    
    def initialize(self, s0, a0):
        for theta in self.candidate_thetas:
            inner_particles = []
            likelihoods = []
            o0 = s0.copy()
            m0 = s0.copy()
            b0 = s0.copy()
            for _ in range(self.M):
                lkh = self.compute_likelihood(b0, a0)
                inner_particles.append({'m': m0, 'b': b0, 'weight': lkh})
                likelihoods.append(lkh)
            likelihoods = np.array(likelihoods)
            norm_weights = likelihoods / np.sum(likelihoods) if np.sum(likelihoods) > 0 else np.ones(self.M)/self.M
            for j in range(self.M):
                inner_particles[j]['weight'] = norm_weights[j]
            self.theta_particles[theta]['inner_particles'] = inner_particles
            self.theta_particles[theta]['theta_weight'] = np.mean(likelihoods)
        all_weights = np.array([self.theta_particles[theta]['theta_weight'] for theta in self.candidate_thetas])
        norm_all = all_weights / np.sum(all_weights)
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
            for particle in inner_particles:
                o_new = self.sample_observation(s)
                m_new = self.sample_memory(particle['m'], o_new, theta)
                b_new = self.update_belief(m_new)
                lkh = self.compute_likelihood(b_new, a)
                particle['m'] = m_new
                particle['b'] = b_new
                particle['weight'] *= lkh
                likelihoods.append(lkh)
            weights = np.array([p['weight'] for p in inner_particles])
            norm_weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(self.M)/self.M
            for idx, particle in enumerate(inner_particles):
                particle['weight'] = norm_weights[idx]
            avg_L = np.mean(likelihoods)
            self.theta_particles[theta]['theta_weight'] *= avg_L
        candidate_weights = np.array([self.theta_particles[theta]['theta_weight'] for theta in self.candidate_thetas])
        candidate_weights = candidate_weights / np.sum(candidate_weights) if np.sum(candidate_weights) > 0 else np.ones(len(self.candidate_thetas))/len(self.candidate_thetas)
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

    def sample_memory(self, m, o, theta):
        return self.agent.sample_memory(m, o, theta)

    def update_belief(self, m):
        return self.agent.update_belief(m)

    def compute_likelihood(self, b, a):
        return self.agent.compute_policy_probs(b)[list(self.agent.action_space.keys())[a]]


