import copy
import numpy as np
from feature import *

# Actor base class
class Actor:

    def __init__(self, env, eta):

        self.env = env

        # step size for functional update (surrogate construction)
        self.eta = eta

    def update_policy_param():
        pass

    def update_inner_weight():
        pass

    def get_policy_prob():
        pass

    def get_objective():
        pass


# Tabular Actor with NPG/MDPO update. (MDPO in direct tabular case is equivalent to NPG).
class DirectTabularNPGActor(Actor):
    def __init__(self, env, eta, init_theta):
        super().__init__(env, eta)

        # current theta
        self.current_theta = copy.deepcopy(init_theta)
        self.probs = self.get_policy_prob(self.current_theta)
        self.c = None

    def get_objective():
        pass
    
    # update actor params (theta)
    def update_policy_param(self, Q_pit):
        pr_pit = copy.deepcopy(self.current_theta)
        current_theta = self.update_inner_weight(pr_pit, Q_pit)
        self.current_theta = current_theta
        self.probs = self.get_policy_prob(self.current_theta)

    
    # based on c_in_stepsize, set the effective step-size
    def set_effective_eta(self):
        if self.c is None:
            etap = self.eta
        else:
            etap = (self.eta * self.c) / (self.eta + self.c)
        return etap
        
    # The update is NPG, which has exact solution.
    def update_inner_weight(self, pr_pit, Q_pit):
        
        d = self.env.calc_dpi(pr_pit)
        etap = self.set_effective_eta()
        eps = 1e-10

        tmp = Q_pit - np.max(Q_pit, axis=1).reshape(-1, 1)
        exp_eta_Q_pit = np.exp(etap * tmp) 
        denom = np.einsum('sa,sa->s', np.einsum('sa,s->sa', pr_pit, d),  exp_eta_Q_pit)
        num = np.einsum('sa,sa->sa',  np.einsum('sa,s->sa', pr_pit, d),  exp_eta_Q_pit)
        return num/(denom.reshape(-1, 1)+eps)
    
    # get probs safely
    def get_policy_prob(self, probs):
        delta = (np.ones(self.env.state_space) - np.sum(probs, axis=1)) / self.env.action_space
        probs += delta.reshape(-1, 1)
        return probs

    # setting c
    def set_c(self, c):
        self.c = c


# Linear actor with direct representation (Linear MDPO)
class DirectLinearMDPOActor(Actor):
    def __init__(self, env, eta, init_theta, d, num_tiles, tiling_size, m, lr, stop_trs):
        super().__init__(env, eta)

        self.tabular_tc = TabularTileCoding(d, num_tiles, tiling_size)
        self.tc_feature = TileCodingFeatures(self.env.action_space, self.tabular_tc.get_tile_coding_args())
        self.feature_dim = d
        self.current_theta = init_theta
        self.features = self.get_features()

        # probability (direct policy) of current_theta
        self.probs = self.get_policy_prob(self.current_theta)
        self.m = m
        self.lr = lr
        self.stop_trs = stop_trs
        self.c = None

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

    # one-hot features (env.S * env.A * feature_dim)
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

    # set the effective step-size
    def set_effective_eta(self):
        if self.c is None:
            etap = self.eta
        else:
            etap = (self.eta * self.c) / (self.eta + self.c)
        return etap

    # actor objective
    def get_objective(self, Q_pit, d_pit, pr_pit, theta):
        probs = self.get_policy_prob(theta)
        mu = np.einsum('s,sa->sa', d_pit, pr_pit)
        eps = 1e-10
        etap = self.set_effective_eta()
        loss = np.sum(mu * ((probs/(pr_pit+1e-10)) * (Q_pit - (1/etap) * np.log(np.maximum(eps, probs/(pr_pit+eps))))))
        return loss
    
    # update params
    def update_policy_param(self, Q_pit):
        pr_pit = self.get_policy_prob(self.current_theta)
        current_theta = self.update_inner_weight(pr_pit, Q_pit)
        self.current_theta = copy.deepcopy(current_theta)
        self.probs = self.get_policy_prob(self.current_theta)
        
    # off-policy updates
    def update_inner_weight(self, pr_pit, Q_pit):

        d_pit = self.env.calc_dpi(pr_pit)
        tmp_theta = copy.deepcopy(self.current_theta)

        etap = self.set_effective_eta()

        grad = [1]
        i = 0
        eps = 1e-10
        lr = self.lr
        while i < self.m and np.linalg.norm(grad) > self.stop_trs:
            probs = self.get_policy_prob(tmp_theta)

            grad1 = self.get_grad_pi_wrt_theta(tmp_theta)
            grad = np.einsum('s,sd->d', d_pit, np.einsum('sad,sa->sd', grad1, Q_pit - (1/etap) * (1 + np.log(np.maximum(eps, probs/(pr_pit+eps))))))

            eta = self.armijo(Q_pit, d_pit, pr_pit, tmp_theta, grad, lr)
            tmp_theta += eta * grad
            
            lr = 1.8 * eta
            i += 1
        return tmp_theta
    
    # get theta and return prob (direct policy)
    def get_policy_prob(self, theta):

        logits = np.einsum('d,sad->sa', theta, self.features)
        max_logits = np.max(logits, axis=1, keepdims=True)

        # Compute the exponentials of the shifted logits and their sum along each row
        exp_shifted = np.exp(logits - max_logits)
        exp_sum = np.sum(exp_shifted, axis=1, keepdims=True)

        # Compute the softmax probabilities using einsum
        probs = np.einsum('sa,sa->sa', exp_shifted, 1 / exp_sum)
        return probs

    # grad p^{\pi} w.r.t. param \theta
    def get_grad_pi_wrt_theta(self, theta):
        probs = self.get_policy_prob(theta)

        logits = np.einsum('d,sad->sa', theta, self.features)
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_shifted = np.exp(logits - max_logits)

        tmp = self.features - ((np.einsum('sa,sad->sd', exp_shifted, self.features)) / ((np.sum(exp_shifted, axis=1)).reshape(-1, 1))).reshape(self.env.state_space, 1, self.feature_dim)
        grad = np.einsum('sa,sad->sad', probs, tmp)
        return grad

    
    # set c
    def set_c(self, c):
        self.c = c

    # armijo for off-policy step-size
    def armijo(self, Q_pit, pr_pit, d_pit, theta, grad, eta_max, beta=0.9, c=0.0001):
        etak = eta_max
        while -1 * self.get_objective(Q_pit, pr_pit, d_pit, theta + etak*grad) > -1 * self.get_objective(Q_pit, pr_pit, d_pit, theta) - c*etak* np.linalg.norm(grad)**2:
            etak *= beta
        return etak
    

# Tabular actor with sMDPO update.
class SoftmaxTabularsMDPOActor(Actor):
    def __init__(self, env, eta, init_theta):
        super().__init__(env, eta)

        # current theta
        self.current_theta = copy.deepcopy(init_theta)
        self.probs = self.get_policy_prob(self.current_theta)
        self.c = None
    
    
    def update_policy_param(self, A_pit):
        pr_pit = copy.deepcopy(self.current_theta)
        current_theta = self.update_inner_weight(pr_pit, A_pit)
        self.current_theta = current_theta
        self.probs = self.get_policy_prob(self.current_theta)
        

    # p^{\pi}(a|s) = p^{\pi}_t (a|s) max(1 + \eta * A^{\pi}_t(s, a), 0) / \sum_b p^{\pi}_t(b|s) max(1 + \eta * A^{\pi}_t(s, b), 0)}
    def update_inner_weight(self, pr_pit, A_pit):
        eps = 1e-8

        etap = self.set_effective_eta()
        d = self.env.calc_dpi(pr_pit)
        
        max_eta_A = np.where(1 + etap * A_pit > eps, 1 + etap * A_pit, eps)
        denom = np.einsum('sa,sa->s', np.einsum('sa,s->sa', pr_pit, d), max_eta_A)
        num = np.einsum('sa,sa->sa', np.einsum('sa,s->sa', pr_pit, d), max_eta_A)

        return num/(denom.reshape(-1, 1) + eps)
    
    # get probs safely
    def get_policy_prob(self, probs):
        delta = (np.ones(self.env.state_space) - np.sum(probs, axis=1)) / self.env.action_space
        probs += delta.reshape(-1, 1)
        return probs


    
    # set the effective step-size
    def set_effective_eta(self):
        if self.c is None:
            etap = self.eta
        else:
            etap = (self.eta * self.c) / (self.eta + self.c)
        return etap
        

    def set_c(self, c):
        self.c = c


# Linear actor with sMDPO update.
class SoftmaxLinearsMDPOActor(Actor):
    def __init__(self, env, eta, init_theta, d, num_tiles, tiling_size, m, lr, stop_trs):
        super().__init__(env, eta)

        self.tabular_tc = TabularTileCoding(d, num_tiles, tiling_size)
        self.tc_feature = TileCodingFeatures(self.env.action_space, self.tabular_tc.get_tile_coding_args())
        self.feature_dim = d
        #policy parameters
        self.current_theta = init_theta
        # self.features = self.get_tabular_features()
        self.features = self.get_features()

        self.probs = self.get_policy_prob(self.current_theta)
        self.m = m
        self.lr = lr
        self.stop_trs = stop_trs
        self.c = None
    
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

    # based on c_in_stepsize, set the effective step-size
    def set_effective_eta(self):
        if self.c is None:
            etap = self.eta
        else:
            etap = (self.eta * self.c) / (self.eta + self.c)
        return etap
    
    # actor objective
    def get_objective(self, A_pit, d_pit, pr_pit, theta):
        probs = self.get_policy_prob(theta)
        mu = np.einsum('s,sa->sa', d_pit, pr_pit)
        eps = 1e-10
        etap = self.set_effective_eta()
        loss = np.sum(mu * ((A_pit + 1/etap) * np.log(np.maximum(eps, probs/(pr_pit+eps)))))
        return loss
    
    # update actor params
    def update_policy_param(self, A_pit):
        pr_pit = self.get_policy_prob(self.current_theta)
        current_theta = self.update_inner_weight(pr_pit, A_pit)
        self.current_theta = copy.deepcopy(current_theta)
        self.probs = self.get_policy_prob(self.current_theta)
        
    # off-policy update
    def update_inner_weight(self, pr_pit, A_pit):

        etap = self.set_effective_eta()
        d_pit = self.env.calc_dpi(pr_pit)
        tmp_theta = copy.deepcopy(self.current_theta)
        grad = [1]
        i = 0
        lr = self.lr
        eps = 1e-10

        while i < self.m and np.linalg.norm(grad) > self.stop_trs:
            probs = self.get_policy_prob(tmp_theta)
            grad1 = self.get_grad_pi_wrt_theta(tmp_theta)
            mu = np.einsum('s,sa->sa', d_pit, pr_pit)
            grad = np.einsum('sa,sad->d', mu, np.einsum('sa,sad->sad', A_pit + 1/etap, grad1/np.expand_dims((probs+eps), axis=2)))
            eta = self.armijo(A_pit, d_pit, pr_pit, tmp_theta, grad, lr)
            lr = 1.8 * eta
            tmp_theta += eta * grad
            i += 1
        return tmp_theta
    
    # get direct representation (prob) of policy.
    def get_policy_prob(self, theta):
        logits = np.einsum('d,sad->sa', theta, self.features)
        max_logits = np.max(logits, axis=1, keepdims=True)

        # Compute the exponentials of the shifted logits and their sum along each row
        exp_shifted = np.exp(logits - max_logits)
        exp_sum = np.sum(exp_shifted, axis=1, keepdims=True)

        # Compute the softmax probabilities using einsum
        probs = np.einsum('sa,sa->sa', exp_shifted, 1 / exp_sum)

        return probs

    # grad of policy w.r.t. actor param
    def get_grad_pi_wrt_theta(self, theta):
        probs = self.get_policy_prob(theta)

        logits = np.einsum('d,sad->sa', theta, self.features)
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_shifted = np.exp(logits - max_logits)

        tmp = self.features - ((np.einsum('sa,sad->sd', exp_shifted, self.features)) / ((np.sum(exp_shifted, axis=1)).reshape(-1, 1))).reshape(self.env.state_space, 1, self.feature_dim)
        grad = np.einsum('sa,sad->sad', probs, tmp)
        return grad

    def set_c(self, c):
        self.c = c

    # Armijo
    def armijo(self, A_pit, pr_pit, d_pit, theta, grad, eta_max, beta=0.9, c=0.0001):
        etak = eta_max
        while -1 * self.get_objective(A_pit, pr_pit, d_pit, theta + etak*grad) > -1 * self.get_objective(A_pit, pr_pit, d_pit, theta) - c*etak* np.linalg.norm(grad)**2:
            etak *= beta
        return etak
