import numpy as np
import multiprocessing as mp
from environments.TabularMDPs import *
import random
import statistics

class Sampling:

    def __init__(self, env, num_samples, ep_len):
        self.env = env
        self.num_samples = num_samples
        self.ep_len = ep_len
        self.S, self.A = self.env.state_space, self.env.action_space
        self.r = self.env.r
        self.P = self.env.P
        self.gamma = self.env.gamma

    def get_data(self):
        pass


class MCSampling(Sampling):
    """
    Monte Carlo Sampling
    """

    def __init__(self, env, num_samples, ep_len):
        super().__init__(env, num_samples, ep_len)
        self.policy_prob = None

    # run a single trajectory starting from state 's' action 'a' and output return 'G'
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
    
    # get Q for all states and actions.
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


class TDSampling(Sampling):
    """
    Temporal Difference (0) Sampling
    """
    def __init__(self, env, num_samples, ep_len):
        super().__init__(env, num_samples, ep_len)
        self.policy_prob = None

    # return pairs of (SARSA) starting at state 's' and action 'a'
    def rollout(self, sa):
        s,a = sa
        data = []
        for k in range(self.ep_len):
            next_s = np.random.choice(self.S, p=self.P[s, a])
            next_a = np.random.choice(self.A, p=self.policy_prob[next_s])
            if self.r.ndim == 2:
                r =  self.r[s, a]
            elif self.r.ndim == 3:
                r = self.r[s, a, next_s]
            data.append(Data(s, a, r, next_s, next_a))
            s, a = next_s, next_a
        return data
    
    # collect data (SARSA) for TD update 
    def get_data(self, policy_prob):
        self.set_policy_prob(policy_prob)
        random.seed(42)
        pairs = [(random.choice(np.arange(self.S)), random.choice(np.arange(self.A))) for i in range(self.num_samples)]
        num_processes = mp.cpu_count()
        with mp.Pool(num_processes) as executor:
            data = executor.map(self.rollout, pairs)
        data = np.array(data).reshape(-1, )
        return data
        
    def set_policy_prob(self, policy_prob):
        self.policy_prob = policy_prob

    class Data:
        def __init__(self, s, a, r, next_s, next_a):
            self.s = s
            self.a = a
            self.r = r
            self.next_s = next_s
            self.next_a = next_a
    




















class LSTDContinuousSampling:

    def __init__(self, env, tc_feature, num_traj, seed=1):
        self.env = env
        self.num_traj = num_traj
        self.rng = np.random.RandomState(seed)
        self.tc_feature = tc_feature
        self.num_features = self.tc_feature.get_feature_size()

    def get_probs(self, theta, state):      
        d = self.tc_feature.get_feature_size()
        logits = []
        for a in range(self.env.A):
            feature = self.tc_feature.one_hot(d, self.tc_feature.get_feature(state, a))
            logits.append(np.einsum('d,d', feature, theta))
        e_logits = np.exp(logits - np.max(logits))
        probs = e_logits/e_logits.sum(0)

        return probs


    def get_data(self, theta):

        data = []
        states = []
        return_value = 0
        for t in range(self.num_traj):
            state = self.env.get_init_state()
            probs = self.get_probs(theta, state)
            action = self.rng.choice(self.env.A, p=probs)
            running = True
            n_step = 0
            return_traj = 0.0
            current_discount = 1.0
            while running:
                states.append(state)
                next_state, reward = self.env.step(state, action)
                if n_step == self.env._max_steps or self.env.is_state_over_bounds(next_state):
                    running = False
                    continue
                probs = self.get_probs(theta, next_state)
                next_action = self.rng.choice(self.env.A, p=probs)
                sample_data = Data(state, action, next_state, next_action, reward)
                data.append(sample_data)
                return_traj += current_discount * reward
                current_discount *= self.env.gamma
                state = copy.deepcopy(next_state)
                action = copy.deepcopy(next_action)
                n_step += 1
            return_value += return_traj
        print(f' J {return_value / self.num_traj}')
        return data

class Data:
    def __init__(self, s, a, next_s, next_a, r):
        self.s = s
        self.a = a
        self.next_s = next_s
        self.next_a = next_a
        self.r = r

if __name__ == "__main__":
    env = get_CP()
    num_traj = 100
    seed = 1
    sampler = MCContinuousSampling(env, num_traj, seed)
    policy = None
    Q = sampler.get_data(policy)
    print(Q)

