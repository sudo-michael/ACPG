import numpy as np
from feature import *
import copy
import matplotlib.pyplot as plt
import time

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


# class DirectLFABoostTDCritic(Critic):
#     def __init__(self, env, d, num_tiles, tiling_size, lr):
#         super().__init__(env)
#         self.tabular_tc = TabularTileCoding(d, num_tiles, tiling_size)
#         self.tc_feature = TileCodingFeatures(self.env.action_space, self.tabular_tc.get_tile_coding_args())
#         self.feature_dim = d
#         self.theta = np.random.normal(0, 1, self.feature_dim)
#         self.gamma = self.env.gamma
#         self.lr = lr
#         # self.features = self.get_features()
#         self.features = self.get_tabular_features()


#     def get_features(self):
#         S = self.env.state_space
#         A = self.env.action_space
#         features = np.zeros((S, A, self.feature_dim))
#         for s in range(S):
#             for a in range(A):
#                 not_one_zero = self.tc_feature.get_feature([s], a)
#                 one_zero = np.zeros(self.feature_dim)
#                 one_zero[not_one_zero] = 1
#                 features[s, a, :] = one_zero
#         return features

    
#     # one-hot features
#     def get_tabular_features(self):
#         S = self.env.state_space
#         A = self.env.action_space
#         features = np.zeros((S, A, self.feature_dim))
#         for s in range(S):
#             for a in range(A):
#                 one_zero = np.zeros(self.feature_dim)
#                 one_zero[A*s + a] = 1
#                 features[s, a, :] = one_zero
#         return features 

#     # get Q_hat
#     def get_estimated_Q(self, data):
#         theta = self.compute_theta(data)
#         Q_hat_pit = np.einsum('sad,d->sa', self.features, theta)
#         # print(Q_hat_pit)
#         return Q_hat_pit

#     # solve critic param \theta
#     def compute_theta(self, data):
#         theta = copy.deepcopy(self.theta)
#         for d in data:
#             s, a, r, next_s, next_a = d.s, d.a, d.r, d.next_s, d.next_a
#             # print(s, a, r, next_s, next_a)
#             Q_next_s_next_a = np.einsum('d,d->', theta, self.features[next_s, next_a, :])
#             Q_s_a = np.einsum('d,d->', theta, self.features[s, a, :])
#             theta += self.lr * (r + self.gamma * Q_next_s_next_a - Q_s_a) * self.features[s, a, :]
#         self.theta = copy.deepcopy(theta)
#         return self.theta


# Linear parameterized critic that estimates Q with squared loss.
class DirectLFATDCritic(Critic):
    def __init__(self, env, d, num_tiles, tiling_size):
        super().__init__(env)

        self.tabular_tc = TabularTileCoding(d, num_tiles, tiling_size)
        self.tc_feature = TileCodingFeatures(self.env.action_space, self.tabular_tc.get_tile_coding_args())
        self.feature_dim = d
        self.features = self.get_features()
        # self.features = self.get_tabular_features()
        self.theta = np.random.normal(0, 0.1, d)

    # tile-coded features
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
        
    # one-hot features
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

    # get Q_hat
    def get_estimated_Q(self, Q_pit, pr_pit, d_pit):
        theta = self.compute_theta(Q_pit, pr_pit, d_pit)
        self.theta = copy.deepcopy(theta) # warmup
        Q_hat_pit = np.einsum('sad,d->sa', self.features, self.theta)
        return Q_hat_pit

    # solve critic param \theta
    def compute_theta(self, Q_pit, pr_pit, d_pit):


        mu_pit_flatten = np.einsum('sa,s->sa', pr_pit, d_pit).flatten()
        phi_flatten = self.features.reshape(-1, self.features.shape[-1])
        Q_pit_flatten = Q_pit.flatten()
        y = np.einsum('ds,s->d', mu_pit_flatten * phi_flatten.T, Q_pit_flatten)
        A = np.einsum('ds,sm->dm', mu_pit_flatten * phi_flatten.T, phi_flatten)
        # handle the SVD issue
        try:
            theta = np.linalg.lstsq(A, y, rcond=None)[0]
        except np.linalg.LinAlgError as e:
            theta = np.linalg.lstsq(A+np.random.normal(0, 0.1, A.shape), y, rcond=None)[0]
        return theta

        # theta = copy.deepcopy(self.theta)
        # grad = np.array([1])
        # lr = self.lr
        # i = 0
        # while np.linalg.norm(grad) > self.stop_trs and i < self.m:
        #     grad = self.get_gradient(Q_pit, pr_pit, d_pit, theta)
        #     eta = self.armijo(Q_pit, pr_pit, d_pit, theta, grad, lr)
        #     theta -= eta * grad
        #     lr = 1.8*eta # a trick to accelerate armijo computation
        #     i += 1
        # print(i)
        # return theta
        

    def get_gradient(self, Q_pit, pr_pit, d_pit, theta):
        mu_pit = np.einsum('sa,s->sa', pr_pit, d_pit)
        Q_hat_pit = np.einsum('sad,d->sa', self.features, theta)
        tmp1 = np.transpose(self.features, (2, 0, 1))
        grad = 2 * np.einsum('dsa,sa->d', tmp1, (Q_hat_pit - Q_pit) * mu_pit)
        return grad


    # get critic objective
    def get_loss(self, Q_pit, pr_pit, d_pit, theta):
        mu_pit = np.einsum('sa,s->sa', pr_pit, d_pit)
        Q_hat_pit = np.einsum('sad,d->sa', self.features, theta)
        loss = np.dot(mu_pit.flatten(), ((Q_hat_pit - Q_pit)**2).flatten())
        return loss

    # armijo for off-policy step-size
    def armijo(self, Q_pit, pr_pit, d_pit, theta, grad, eta_max, beta=0.9, c=0.0001):
        etak = eta_max
        while self.get_loss(Q_pit, pr_pit, d_pit, theta - etak*grad) > self.get_loss(Q_pit, pr_pit, d_pit, theta) - c*etak* np.linalg.norm(grad)**2:
            etak *= beta
        return etak


# Linear parameterized critic that estimates Q with decision-aware loss
class DirectLFAACPGCritic(Critic):
    def __init__(self, env, d, num_tiles, tiling_size, eta, c, m, lr, stop_trs):
        super().__init__(env)
        
        self.tabular_tc = TabularTileCoding(d, num_tiles, tiling_size)
        self.tc_feature = TileCodingFeatures(self.env.action_space, self.tabular_tc.get_tile_coding_args())
        self.feature_dim = d
        
        self.theta = np.random.normal(0, 0.1, d)
        #self.features = self.get_tabular_features()
        self.features = self.get_features()
        self.eta = eta
        self.c = c

        self.m = m
        self.lr = lr
        self.stop_trs = stop_trs

   
   # tile-coded features
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

    # one-hot features
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

    # get Q_hat 
    def get_estimated_Q(self, Q_pit, pr_pit, d_pit):
        theta = self.compute_theta(Q_pit, pr_pit, d_pit)
        self.theta = copy.deepcopy(theta) # warmup initialization
        Q_hat_pit = np.einsum('sad,d->sa', self.features, self.theta)
        return Q_hat_pit

    # compute params
    def compute_theta(self, Q_pit, pr_pit, d_pit):
        theta = copy.deepcopy(self.theta)
        grad = np.array([1])
        lr = self.lr
        i = 0
        while np.linalg.norm(grad) > self.stop_trs and i < self.m:
            grad = self.get_gradient(Q_pit, pr_pit, d_pit, theta)
            eta = self.armijo(Q_pit, pr_pit, d_pit, theta, grad, lr)
            theta += eta * grad
            lr = 1.8*eta # a trick to accelerate armijo computation
            i += 1
        return theta

    # get grad of critic objective
    def get_gradient(self, Q_pit, pr_pit, d_pit, theta):
        eps = 1e-20
        term1 = np.einsum('sa,sad->sd', pr_pit, self.features)
        Q_hat_pit = np.einsum('sad,d->sa', self.features, theta)
        tmp = (Q_hat_pit - Q_pit) - np.max(Q_hat_pit - Q_pit, axis = 1).reshape(-1, 1)
        t2num = np.einsum('sa,sad->sd', np.multiply(pr_pit, np.exp(self.c * tmp)), self.features)
        t2den = np.einsum('sa,sa->s', pr_pit, np.exp(self.c * tmp))
        term2 = t2num / (t2den.reshape(-1, 1)+eps)
        grad = np.einsum('s,sd->d', d_pit, term1-term2)
        return grad

    # get critic objective
    def get_loss(self, Q_pit, pr_pit, d_pit, theta):
        eps = 1e-20
        Q_hat_pit = np.einsum('sad,d->sa', self.features, theta)
        term1 = np.einsum('sa,sa->s', pr_pit, (Q_hat_pit - Q_pit))
        tmp = (Q_hat_pit - Q_pit) - np.max(Q_hat_pit - Q_pit, axis = 1).reshape(-1, 1)
        maximum = np.max(Q_hat_pit - Q_pit, axis = 1)
        term2 = 1/self.c * (np.log(np.einsum('sa,sa->s', pr_pit, np.exp(self.c *(tmp))) + eps) + self.c * maximum)
        loss = np.dot(d_pit, term1-term2)
        return loss
    

    # armijo for off-policy step-size
    def armijo(self, Q_pit, pr_pit, d_pit, theta, grad, eta_max, beta=0.9, c=0.0001):
        etak = eta_max
        while -1 * self.get_loss(Q_pit, pr_pit, d_pit, theta + etak*grad) > -1 * self.get_loss(Q_pit, pr_pit, d_pit, theta) - c*etak* np.linalg.norm(grad)**2:
            etak *= beta
        return etak


# Linear parametrized critic that estimates A with squared loss.
class SoftmaxLFATDCritic(Critic):
    
    def __init__(self, env, d, num_tiles, tiling_size):
        super().__init__(env)

        self.tabular_tc = TabularTileCoding(d, num_tiles, tiling_size)
        self.tc_feature = TileCodingFeatures(self.env.action_space, self.tabular_tc.get_tile_coding_args())
        self.feature_dim = d
        self.features = self.get_features()
        self.c = None
        # self.features = self.get_tabular_features()

    # tile-coded featiures
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

    # one-hot features
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
    
    #get Q_hat 
    def get_estimated_Q(self, A_pit, pr_pit, d_pit):
        theta = self.compute_theta(A_pit, pr_pit, d_pit)
        Q_hat_pit = np.einsum('sad,d->sa', self.features, theta)
        return Q_hat_pit

    #get A_hat 
    def get_estimated_A(self, A_pit, pr_pit, d_pit):
        theta = self.compute_theta(A_pit, pr_pit, d_pit)
        Q_hat_pit = np.einsum('sad,d->sa', self.features, theta)
        A_hat_pit = Q_hat_pit - np.einsum('sa,sa->s', pr_pit, Q_hat_pit).reshape(-1, 1)
        return A_hat_pit

    # calc critic param
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
        try:
            theta = np.linalg.lstsq(A, y, rcond=None)[0]
        except np.linalg.LinAlgError as e:
            theta = np.linalg.lstsq(A+np.random.normal(0, 0.1, A.shape), y, rcond=None)[0]
        return theta

    # Squared loss on advantage
    def get_loss(self, A_pit, pr_pit, d_pit, theta):
        Q_hat_pit = np.einsum('sad,d->sa', self.features, theta)
        V_hat_pit = np.einsum('sa,sa->s', pr_pit, Q_hat_pit)
        A_hat_pit = Q_hat_pit - V_hat_pit.reshape(-1, 1)
        term = (A_hat_pit - A_pit)**2
        loss = np.sum(np.multiply(np.einsum('s,sa->sa', d_pit, pr_pit), term))
        return loss



# Linear parametrized critic that estimates A with our loss.
class SoftmaxLFAACPGCritic(Critic):
    def __init__(self, env, d, num_tiles, tiling_size, eta, c, m, lr, stop_trs):
        super().__init__(env)
        self.tabular_tc = TabularTileCoding(d, num_tiles, tiling_size)
        self.tc_feature = TileCodingFeatures(self.env.action_space, self.tabular_tc.get_tile_coding_args())
        self.feature_dim = d
        self.theta = np.random.normal(0, 0.1, d)
        # self.features = self.get_tabular_features()
        self.features = self.get_features()
        self.eta = eta
        self.c = c 
        self.m = m 
        self.lr = lr 
        self.stop_trs = stop_trs

    # tile-coded features
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

    # one-hot features
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

    # get A_hat
    def get_estimated_A(self, A_pit, pr_pit, d_pit):
        theta = self.compute_theta(A_pit, pr_pit, d_pit)
        self.theta = copy.deepcopy(theta)
        Q_hat_pit = np.einsum('sad,d->sa', self.features, self.theta)
        V_hat_pit = np.einsum('sa,sa->s', pr_pit, Q_hat_pit)
        A_hat_pit = Q_hat_pit - V_hat_pit.reshape(-1, 1)
        return A_hat_pit

    # update critic params
    def compute_theta(self, A_pit, pr_pit, d_pit):

        theta = copy.deepcopy(self.theta)
        grad = np.array([1])
        lr = self.lr
        i = 0
        while np.linalg.norm(grad) > self.stop_trs and i < self.m:
            grad = self.get_gradient(A_pit, pr_pit, d_pit, theta)
            armijoeta = self.armijo(A_pit, pr_pit, d_pit, theta, grad, lr) #for cw
            lr = 1.8*armijoeta # a trick to accelerate armijo computation for cw
            theta += armijoeta * grad
            i += 1
        return theta

    # gradient of critic objective w.r.t. param
    def get_gradient(self, A_pit, pr_pit, d_pit, theta):

        Q_hat_pit = np.einsum('sad,d->sa', self.features, theta)
        V_hat_pit = np.einsum('sa,sa->s', pr_pit, Q_hat_pit)
        A_hat_pit = Q_hat_pit - V_hat_pit.reshape(-1, 1)

        eps = 1e-10
        term1 = np.log(np.maximum(eps, 1 - self.c * (A_pit - A_hat_pit)))
        
        featuresum = np.einsum('sa,sad->sd', pr_pit, self.features).reshape((self.env.state_space, 1, self.feature_dim))
        term2 = self.features - featuresum

        term1dotterm2 = np.einsum('sa,sad->sad', term1, term2)
        
        mu = np.einsum('s,sa->sa', d_pit, pr_pit)

        grad = -1 * np.einsum('sa,sad->d', mu, term1dotterm2)
        # print("critic grad", np.linalg.norm(grad))
        return grad

    # critic objective
    def get_loss(self, A_pit, pr_pit, d_pit, theta):
        Q_hat_pit = np.einsum('sad,d->sa', self.features, theta)
        V_hat_pit = np.einsum('sa,sa->s', pr_pit, Q_hat_pit)
        A_hat_pit = Q_hat_pit - V_hat_pit.reshape(-1, 1)
        eps = 1e-10
        term1 = np.maximum(eps, 1 - self.c * ((A_pit - A_hat_pit)))
        term = np.multiply(term1, np.log(term1))
        loss = np.sum(np.multiply(np.einsum('s,sa->sa', -1/self.c * d_pit, pr_pit), term))
        return loss

    def set_c(self, c):
        self.c = c

    # armijo for inner loop stepsize
    def armijo(self, A_pit, pr_pit, d_pit, theta, grad, eta_max, beta=0.9, step=1e-4):
        etak = eta_max
        while -1 * self.get_loss(A_pit, pr_pit, d_pit, theta + etak*grad) > -1 * self.get_loss(A_pit, pr_pit, d_pit, theta) - step*etak* np.linalg.norm(grad)**2:
            etak *= beta
        return etak













        





        
    
        


