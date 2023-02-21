import numpy as np
from ACPG.feature import *
import copy
import matplotlib.pyplot as plt

class Critic:

    def __init__(self, env):

        self.env = env

    def get_estimated_Q():
        pass


class RandomCritic(Critic):
    def __init__(self, env, epsilon):
        super().__init__(env)
        self.epsilon = epsilon


    def get_estimated_Q(self, Q_pit):
        return Q_pit + np.random.normal(0, self.epsilon, size=Q_pit.shape)

class LFATDCritic(Critic):

    def __init__(self, env, iht_size, num_tiles, tiling_size):
        super().__init__(env)

        self.tabular_tc = TabularTileCoding(iht_size, num_tiles, tiling_size)
        self.tc_feature = TileCodingFeatures(self.env.action_space, self.tabular_tc.get_tile_coding_args())
        self.feature_dim = iht_size
        self.features = self.get_features()
    
    def get_estimated_Q(self, Q_pit):
        theta = self.compute_theta(Q_pit)
        Q_hat_pit = np.einsum('sad,d->sa', self.features, theta)
        return Q_hat_pit

    def compute_theta(self, Q_pit):
        A = self.features.reshape(-1, self.features.shape[-1])
        y = Q_pit.flatten()
        theta = np.linalg.lstsq(A, y, rcond=None)[0]
        return theta

    def get_features(self):
        S = self.env.state_space
        A = self.env.action_space
        features = np.zeros((S, A, self.feature_dim))
        for s in range(S):
            for a in range(A):
                not_one_zero = self.tc_feature.get_feature([s], a)
                one_zero = np.zeros(self.feature_dim)
                one_zero[not_one_zero] = 1
                features[s, a, :] = one_zero
        return features

    def get_tabular_features(self):
        S = self.env.state_space
        A = self.env.action_space
        features = np.zeros((S, A, self.feature_dim))
        for s in range(S):
            for a in range(A):
                one_zero = np.zeros(self.feature_dim)
                one_zero[A*s + a] = 1
                features[s, a, :] = one_zero
        return features 


class LFAACPGCritic(Critic):
    def __init__(self, env, iht_size, num_tiles, tiling_size, eta, c, lr):
        super().__init__(env)
        self.tabular_tc = TabularTileCoding(iht_size, num_tiles, tiling_size)
        self.tc_feature = TileCodingFeatures(self.env.action_space, self.tabular_tc.get_tile_coding_args())
        self.feature_dim = iht_size
        self.theta = np.random.normal(0, 0, iht_size)
        #self.features = self.get_tabular_features()
        self.features = self.get_features()
        self.eta = eta
        self.c = c
        self.lr = lr

    def get_estimated_Q(self, Q_pit, pr_pit, d_pit):
        theta = self.compute_theta(Q_pit, pr_pit, d_pit)
        self.theta = copy.deepcopy(theta) # warmup initialization
        Q_hat_pit = np.einsum('sad,d->sa', self.features, self.theta)
        return Q_hat_pit

    def compute_theta(self, Q_pit, pr_pit, d_pit):
        theta = copy.deepcopy(self.theta)
        grad = np.array([1])
        while np.linalg.norm(grad) > 1e-6:
            grad = self.get_gradient(Q_pit, pr_pit, d_pit, theta)
            eta = self.armijo(Q_pit, pr_pit, d_pit, theta, grad, self.lr)
            self.lr = 1.8*eta # a trick to accelerate armijo computation
            theta += eta * grad
        return theta

    def get_gradient(self, Q_pit, pr_pit, d_pit, theta):
        term1 = np.einsum('sa,sad->sd', pr_pit, self.features)

        Q_hat_pit = np.einsum('sad,d->sa', self.features, theta)
        tmp = (Q_hat_pit - Q_pit) - np.max(Q_hat_pit - Q_pit, axis = 1).reshape(-1, 1)
        t2num = np.einsum('sa,sad->sd', np.multiply(pr_pit, np.exp(self.c * tmp)), self.features)
        t2den = np.einsum('sa,sa->s', pr_pit, np.exp(self.c * tmp))
        term2 = t2num / (t2den.reshape(-1, 1))

        grad = np.einsum('s,sd->d', d_pit, term1-term2)
        
        return grad

    def get_features(self):
        S = self.env.state_space
        A = self.env.action_space
        features = np.zeros((S, A, self.feature_dim))
        for s in range(S):
            for a in range(A):
                not_one_zero = self.tc_feature.get_feature([s], a)
                one_zero = np.zeros(self.feature_dim)
                one_zero[not_one_zero] = 1
                features[s, a, :] = one_zero
        return features

    def get_tabular_features(self):
        S = self.env.state_space
        A = self.env.action_space
        features = np.zeros((S, A, self.feature_dim))
        for s in range(S):
            for a in range(A):
                one_zero = np.zeros(self.feature_dim)
                one_zero[A*s + a] = 1
                features[s, a, :] = one_zero
        return features 


    def get_loss(self, Q_pit, pr_pit, d_pit, theta):
        Q_hat_pit = np.einsum('sad,d->sa', self.features, theta)
        tmp = (Q_hat_pit - Q_pit) - np.max(Q_hat_pit - Q_pit, axis = 1).reshape(-1, 1)
        maximum = np.max(Q_hat_pit - Q_pit, axis = 1)
        term1 = np.einsum('sa,sa->s', pr_pit, (Q_hat_pit - Q_pit))
        term2 = 1/self.c * (np.log(np.einsum('sa,sa->s', pr_pit, np.exp(self.c *(tmp)))) + self.c * maximum)
        loss = np.dot(d_pit, term1-term2)
        return loss

    def armijo(self, Q_pit, pr_pit, d_pit, theta, grad, eta_max, beta=0.9, c=0.0001):
        etak = eta_max
        while -1 * self.get_loss(Q_pit, pr_pit, d_pit, theta + etak*grad) > -1 * self.get_loss(Q_pit, pr_pit, d_pit, theta) - c*etak* np.linalg.norm(grad)**2:
            etak *= beta
        return etak







        





        
    
        


