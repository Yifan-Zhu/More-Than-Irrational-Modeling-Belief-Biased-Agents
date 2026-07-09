from enum import IntEnum
import copy
import numpy as np
import gymnasium as gym

class Actions(IntEnum):
    up = 0
    right = 1
    down = 2
    left = 3

class Belief:
    def __init__(self, env: gym.Env, tgt_prior=None):
        self.env = env
        self.grid_size = env.grid_size
        self.num_tgt = env.num_tgt
        self.pos = np.ones((self.grid_size, self.grid_size)) / (self.grid_size * self.grid_size)
        if tgt_prior is None:
            self.tgt_prior = np.ones((1, self.num_tgt)) / self.num_tgt
        else:
            self.tgt_prior = np.array(tgt_prior)
        self.tgt = copy.deepcopy(self.tgt_prior)
        self.obs2pos = self._gen_obs2pos()
        

    def _to_nested_tuple(self, arr):
        if isinstance(arr, list):
            return tuple(self._to_nested_tuple(x) for x in arr)
        return arr 

    def _gen_obs2pos(self):
        """
        The same obs could be generated from multiple positions.
        """
        obs2pos = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                possible_pos = [(i,j)]
                obs = self.env.gen_obs_grid(i,j)
                tuple_obs = self._to_nested_tuple(obs.tolist())
                if tuple_obs not in obs2pos:
                    obs2pos[tuple_obs] = possible_pos
                else:
                    obs2pos[tuple_obs].append((i,j))
        return obs2pos
    
    def reset(self):
        self.pos = np.ones((self.grid_size, self.grid_size)) / (self.grid_size * self.grid_size)
        self.tgt = copy.deepcopy(self.tgt_prior)


    def update(self, memory_decay_model, **kwargs):
        """
        The current update is full-Bayesian instead of Bayesian filtering
        """
        self.reset()
        new_history = memory_decay_model(**kwargs)
        self._update_pos(new_history)
        self._update_tgt(new_history)

    def compute(self, history):
        """
        The current compute is full-Bayesian instead of Bayesian filtering
        """
        self.reset()
        self._update_pos(history)
        self._update_tgt(history)
        return self.get_belief()

    def _update_pos(self, history: list):
        # current belief: self.pos
        pos_belief = self.pos
        for hist in history:
            obs = hist["obs"]['surrounding']
            act = hist["action"]
            next_belief = np.zeros_like(pos_belief)
            # iterate over all possible previous states, i.e. to compute \sum_{s_{t-1}}(p(s_t|s_{t-1},a)b_{t-1}(s_{t-1}))
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    prev_belief = pos_belief[i,j]
                    if prev_belief < 1e-15:
                        continue
                    prob_trans = self.env.transition_prob(i, j, act)
                    for (x,y,prob) in prob_trans:
                        next_belief[x,y] += prev_belief * prob
            lkh_obs = self._compute_likelihood_pos(obs)
            next_belief *= lkh_obs
            sum_next_belief = np.sum(next_belief)
            if sum_next_belief < 1e-12:
                next_belief = np.ones_like(next_belief)
                sum_next_belief = next_belief.size
            next_belief /= sum_next_belief
            pos_belief = next_belief
        self.pos = pos_belief


    def _update_tgt(self, history: list):
        tgt_belief = self.tgt
        for hist in history:
            obs = hist["obs"]['target']
            lkh_obs = self._compute_likelihood_tgt(obs)
            tgt_belief *= lkh_obs
            sum_tgt_belief = np.sum(tgt_belief)
            if sum_tgt_belief < 1e-12:
                tgt_belief = np.ones_like(tgt_belief)
                sum_tgt_belief = tgt_belief.size
            tgt_belief /= sum_tgt_belief
        self.tgt = tgt_belief


    def _compute_likelihood_pos(self, obs):
        """
        Need to consider transition probability distribution, otherwise, there will be an implicit assumption that the agent stands still, and the observation is generated from the same position, however this is not the case, and thus the likelihood will be 0, and will be normalized to uniform distribution.
        """
        tuple_obs = self._to_nested_tuple(obs.tolist())
        likelihood = np.zeros((self.grid_size, self.grid_size))
        if tuple_obs in self.obs2pos:
            possible_pos_list = self.obs2pos[tuple_obs]
            for possible_pos in possible_pos_list:
                x, y = possible_pos
                likelihood[x,y] = 1.0
        else:
            likelihood[:] = 1.0
        return likelihood

    def _compute_likelihood_tgt(self, obs):
        if obs == 0:
            return np.ones((1, self.num_tgt))
        likelihood = np.zeros((1, self.num_tgt))
        likelihood[0, obs-1] = 1.0
        return likelihood

    def get_belief(self):
        return {
            "position": self.pos,
            "target": self.tgt,
        }

def memory_decay_perfect(history: list, **kwargs):
    """
    return perfect record of history
    """
    return history

def memory_decay_rand_drop(history: list, decay: float, **kwargs):
    """
    memory on target observation will be dropped with probability decay * 100%
    """
    new_history = []
    for hist in history:
        if np.random.rand() < decay:
            new_hist = copy.deepcopy(hist)
            new_hist["obs"]["target"] = 0
            new_history.append(new_hist)
        else:
            new_history.append(hist)
    return new_history

