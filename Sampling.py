import numpy as np
import multiprocessing as mp
from ACPG.environments.TabularMDPs import *
import random
import statistics


class Sampling:

    def __init__(self, env, num_samples, ep_len):
        self.env = env
        self.num_samples = num_samples
        self.ep_len = ep_len

    def get_data(self):
        pass


class MCSampling(Sampling):
    """
    Monte Carlo Sampling
    """

    def __init__(self, env, num_samples, ep_len):
        super().__init__(env, num_samples, ep_len)
        self.S, self.A = self.env.state_space, self.env.action_space
        self.r = self.env.r
        self.P = self.env.P
        self.gamma = self.env.gamma
        self.policy_prob = None

    def rollout(self, sa):
        G = 0
        s, a = sa
        for j in range(self.ep_len):
            next_s = np.random.choice(self.S, p=self.P[s, a])
            next_a = np.random.choice(self.A, p=self.policy_prob[next_s])
            if self.r.ndim == 2:
                G += (self.gamma**(j)) * self.r[s, a]
            elif self.r.ndim == 3:
                G += (self.gamma**(j)) * self.r[s, a, next_s]
            s, a = next_s, next_a
        return G
    
    def get_data(self, policy_prob):
        self.set_policy_prob(policy_prob)
        random.seed(42)
        pairs = [(random.choice(np.arange(self.S)), random.choice(np.arange(self.A))) for i in range(self.num_samples)]
        num_processes = mp.cpu_count()
        with mp.Pool(num_processes) as executor:
            Gs = executor.map(self.rollout, pairs)
        aggG = {key: statistics.mean(value for k, value in zip(pairs, Gs) if k == key) for key in set(pairs)}
        Q = np.zeros((self.S, self.A))
        for (row, col), value in aggG.items():
            Q[row, col] = value
        return Q

    def set_policy_prob(self, policy_prob):
        self.policy_prob = policy_prob
    
    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

class Data:
    def __init__(self, s, a, r, next_s, next_a):
        self.s = s
        self.a = a
        self.next_s = next_s
        self.next_a = next_a
        self.r = r

