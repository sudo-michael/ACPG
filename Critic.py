import numpy as np
from ACPG.feature import *
import copy
import matplotlib.pyplot as plt


class Critic:

    def __init__(self, env):

        self.env = env

    def get_estimated_Q():
        pass

# Critic that randomly adds noise to the true Q value.
class RandomCritic(Critic):
    def __init__(self, env, epsilon):
        super().__init__(env)
        self.epsilon = epsilon


    def get_estimated_Q(self, Q_pit):
        return Q_pit + np.random.normal(0, self.epsilon, size=Q_pit.shape)


# Linear parameterized critic that estimates Q with TD loss.
class DirectLFATDCritic(Critic):

    def __init__(self, env, iht_size, num_tiles, tiling_size):
        super().__init__(env)

        self.tabular_tc = TabularTileCoding(iht_size, num_tiles, tiling_size)
        self.tc_feature = TileCodingFeatures(self.env.action_space, self.tabular_tc.get_tile_coding_args())
        self.feature_dim = iht_size
        self.features = self.get_features()
        #self.features = self.get_tabular_features()
    
    def get_estimated_Q(self, Q_pit, pr_pit, d_pit):
        theta = self.compute_theta(Q_pit, pr_pit, d_pit)
        Q_hat_pit = np.einsum('sad,d->sa', self.features, theta)
        return Q_hat_pit

    def compute_theta(self, Q_pit, pr_pit, d_pit):
        mu_pit_flatten = np.einsum('sa,s->sa', pr_pit, d_pit).flatten()
        phi_flatten = self.features.reshape(-1, self.features.shape[-1])
        Q_pit_flatten = Q_pit.flatten()

        y = np.einsum('ds,s->d', mu_pit_flatten * phi_flatten.T, Q_pit_flatten)
        A = np.einsum('ds,sm->dm', mu_pit_flatten * phi_flatten.T, phi_flatten)
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

# Linear parameterized critic that estimates Q with our loss
class DirectLFAACPGCritic(Critic):
    def __init__(self, env, iht_size, num_tiles, tiling_size, eta, c, m, lr):
        super().__init__(env)
        self.tabular_tc = TabularTileCoding(iht_size, num_tiles, tiling_size)
        self.tc_feature = TileCodingFeatures(self.env.action_space, self.tabular_tc.get_tile_coding_args())
        self.feature_dim = iht_size
        self.theta = np.random.normal(0, 0, iht_size)
        #self.features = self.get_tabular_features()
        self.features = self.get_features()
        self.eta = eta
        self.c = c
        self.m = m
        self.lr = lr

    def get_estimated_Q(self, Q_pit, pr_pit, d_pit):
        theta, i = self.compute_theta(Q_pit, pr_pit, d_pit)
        self.theta = copy.deepcopy(theta) # warmup initialization
        Q_hat_pit = np.einsum('sad,d->sa', self.features, self.theta)
        return Q_hat_pit, i

    def compute_theta(self, Q_pit, pr_pit, d_pit):
        theta = copy.deepcopy(self.theta)
        grad = np.array([1])
        lr = self.lr
        i = 0
        while np.linalg.norm(grad) > 1e-5 and i < self.m:
            grad = self.get_gradient(Q_pit, pr_pit, d_pit, theta)
            #print('theta grad', np.linalg.norm(grad))
            eta = self.armijo(Q_pit, pr_pit, d_pit, theta, grad, lr)
            lr = 1.8*eta # a trick to accelerate armijo computation
            theta += eta * grad
            i += 1
        print('critic iter', i)
        return theta, i

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
        term1 = np.einsum('sa,sa->s', pr_pit, (Q_hat_pit - Q_pit))
        tmp = (Q_hat_pit - Q_pit) - np.max(Q_hat_pit - Q_pit, axis = 1).reshape(-1, 1)
        maximum = np.max(Q_hat_pit - Q_pit, axis = 1)
        term2 = 1/self.c * (np.log(np.einsum('sa,sa->s', pr_pit, np.exp(self.c *(tmp)))) + self.c * maximum)
        loss = np.dot(d_pit, term1-term2)
        return loss

    def get_estimated_c(self, Q_pit, pr_pit, d_pit):
        Q_hat_pit = np.einsum('sad,d->sa', self.features, self.theta)
        Q_diff = Q_hat_pit - Q_pit
        tmp = (Q_diff) - np.max(Q_diff, axis = 1).reshape(-1, 1)
        maximum = np.max(Q_diff, axis = 1)
        c = copy.deepcopy(self.c)
        print('first loss', self.get_loss(Q_pit, pr_pit, d_pit, self.theta))
        grad = 1
        for i in range(500):
            term1 = 1/(c**2) * (np.log(np.einsum('sa,sa->s', pr_pit, np.exp(c *tmp))) + c * maximum)
            t2num = np.einsum('sa,sa->s', np.multiply(pr_pit, np.exp(c * tmp)), Q_diff)
            t2den = np.einsum('sa,sa->s', pr_pit, np.exp(c * tmp))
            term2 = 1/c * (t2num / (t2den+1e-10))
            grad = np.dot(d_pit, term1 - term2)
            # print("c grad", grad)
            if c + 0.001 * grad < 0:
                break
            c += 0.001 * grad
            
        self.c = copy.deepcopy(c)
        print('c', c)
        print('second loss', self.get_loss(Q_pit, pr_pit, d_pit, self.theta))
        return c


    def armijo(self, Q_pit, pr_pit, d_pit, theta, grad, eta_max, beta=0.9, c=0.0001):
        etak = eta_max
        while -1 * self.get_loss(Q_pit, pr_pit, d_pit, theta + etak*grad) > -1 * self.get_loss(Q_pit, pr_pit, d_pit, theta) - c*etak* np.linalg.norm(grad)**2:
            etak *= beta
        return etak


# Linear parametrized critic that estimates A with TD loss.
class SoftmaxLFATDCritic(Critic):
    
    def __init__(self, env, iht_size, num_tiles, tiling_size):
        super().__init__(env)

        self.tabular_tc = TabularTileCoding(iht_size, num_tiles, tiling_size)
        self.tc_feature = TileCodingFeatures(self.env.action_space, self.tabular_tc.get_tile_coding_args())
        self.feature_dim = iht_size
        self.features = self.get_features()
        #self.features = self.get_tabular_features()
    
    def get_estimated_A(self, A_pit, pr_pit, d_pit):
        theta = self.compute_theta(A_pit, pr_pit, d_pit)
        Q_hat_pit = np.einsum('sad,d->sa', self.features, theta)
        A_hat_pit = Q_hat_pit - np.einsum('sa,sa->s', pr_pit, Q_hat_pit).reshape(-1, 1)
        return A_hat_pit

    def compute_theta(self, A_pit, pr_pit, d_pit):
        mu_pit_flatten = np.einsum('sa,s->sa', pr_pit, d_pit).flatten() # (SA,1)
        A_pit_flatten = A_pit.flatten() #(SA, 1)
        
        featuresum = np.einsum('sa,sad->sd', pr_pit, self.features).reshape((self.env.state_space, 1, self.feature_dim))
        
        K = (featuresum - self.features).reshape(-1, self.features.shape[-1]) #(SA, d)

        featuresumT = np.einsum('sa,dsa->ds', pr_pit, np.transpose(self.features, (2, 0, 1))).reshape((self.feature_dim, self.env.state_space, 1)) #(d, S, 1)
        KT = (featuresumT - np.transpose(self.features, (2, 0, 1))) # (d, S, A)
        KT = KT.reshape(KT.shape[0], KT.shape[1] * KT.shape[2]) # (d, SA)

        y = np.einsum('ds,s->d', mu_pit_flatten * KT, A_pit_flatten)
        A = -1 * np.einsum('ds,sm->dm', mu_pit_flatten * KT, K)
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


# Linear parametrized critic that estimates A with our loss.
class SoftmaxLFAACPGCritic(Critic):
    def __init__(self, env, iht_size, num_tiles, tiling_size, eta, c, m, lr):
        super().__init__(env)
        self.tabular_tc = TabularTileCoding(iht_size, num_tiles, tiling_size)
        self.tc_feature = TileCodingFeatures(self.env.action_space, self.tabular_tc.get_tile_coding_args())
        self.feature_dim = iht_size
        self.theta = np.random.normal(0, 0.1, iht_size) # it is used to estimate Q.
        #self.features = self.get_tabular_features()
        self.features = self.get_features()
        self.eta = eta
        self.c = c
        self.m = m
        self.lr = lr

    
    def get_estimated_A(self, A_pit, pr_pit, d_pit):
        theta = self.compute_theta(A_pit, pr_pit, d_pit)
        self.theta = copy.deepcopy(theta) # warmup initialization
        Q_hat_pit = np.einsum('sad,d->sa', self.features, self.theta)
        V_hat_pit = np.einsum('sa,sa->s', pr_pit, Q_hat_pit)
        A_hat_pit = Q_hat_pit - V_hat_pit.reshape(-1, 1)
        return A_hat_pit

    # gradient ascent to compute \theta
    def compute_theta(self, A_pit, pr_pit, d_pit):
        theta = copy.deepcopy(self.theta)
        grad = np.array([1])
        lr = self.lr
        i = 0
        while np.linalg.norm(grad) > 1e-6 and i < self.m:
            grad = self.get_gradient(A_pit, pr_pit, d_pit, theta)
            eta = self.armijo(A_pit, pr_pit, d_pit, theta, grad, lr)
            lr = 1.8*eta # a trick to accelerate armijo computation
            theta += eta * grad
            i += 1
        return theta
    
    # gradient of \sum_{s, a} \mu(s, a)||A(s, a) - (\phi(s, a) \theta - \sum_b pr(b|s) \phi(s, a) \theta)||^2 
    # w.r.t. \theta 
    def get_gradient(self, A_pit, pr_pit, d_pit, theta):

        Q_hat_pit = np.einsum('sad,d->sa', self.features, theta)
        V_hat_pit = np.einsum('sa,sa->s', pr_pit, Q_hat_pit)
        A_hat_pit = Q_hat_pit - V_hat_pit.reshape(-1, 1)
        term1 = 1 + np.log(1 - self.c * (A_pit - A_hat_pit))
        
        featuresum = np.einsum('sa,sad->sd', pr_pit, self.features).reshape((self.env.state_space, 1, self.feature_dim))
        term2 = self.features - featuresum

        term1dotterm2 = np.einsum('sa,sad->sad', term1, term2)
        
        mu = np.einsum('s,sa->sa', d_pit, pr_pit)

        grad = -1 * np.einsum('sa,sad->d', mu, term1dotterm2)
        
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


    def get_loss(self, A_pit, pr_pit, d_pit, theta):
        Q_hat_pit = np.einsum('sad,d->sa', self.features, theta)
        V_hat_pit = np.einsum('sa,sa->s', pr_pit, Q_hat_pit)
        A_hat_pit = Q_hat_pit - V_hat_pit.reshape(-1, 1)
        term = np.multiply(1-self.c *(A_pit - A_hat_pit), np.log(np.clip(1 - self.c * (A_pit - A_hat_pit), 1e-2, None)))
        loss = -1/self.c * np.sum(np.multiply(np.einsum('s,sa->sa', d_pit, pr_pit), term))
        return loss

    def armijo(self, A_pit, pr_pit, d_pit, theta, grad, eta_max, beta=0.9, c=0.0001):
        etak = eta_max
        while -1 * self.get_loss(A_pit, pr_pit, d_pit, theta + etak*grad) > -1 * self.get_loss(A_pit, pr_pit, d_pit, theta) - c*etak* np.linalg.norm(grad)**2:
            etak *= beta
        return etak







        





        
    
        


