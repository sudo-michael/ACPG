import copy
import numpy as np


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




# Parametrization \theta is Tabular; a |S*A| matrix storing probs
# Representation \pi is diret; 
# The policy update is NPG
class DirectTabularNPGActor(Actor):
    def __init__(self, env, eta, init_theta):
        super().__init__(env, eta)

        # init_theta is |S*A| matrix with probs
        self.init_theta = copy.deepcopy(init_theta)

        # current theta
        self.current_theta = copy.deepcopy(init_theta)


    def get_objective():
        pass
    
    def update_policy_param(self, Q_pit):
        pr_pit = copy.deepcopy(self.current_theta)
        self.current_theta = self.update_inner_weight(pr_pit, Q_pit)
        

    # The update is NPG. There is not inner loop update
    # p^{\pi}(a|s) = p^{\pi}_t (a|s) e^{\eta * Q^{\pi}_t(s, a)}} / \sum_b p^{\pi}_t(b|s) e^{\eta * Q^{\pi}_t(s, b)}}
    def update_inner_weight(self, pr_pit, Q_pit):
        exp_eta_Q_pit = np.exp(self.eta * Q_pit)
        # denom shape [S]
        denom = np.einsum('sa,sa->s', pr_pit, exp_eta_Q_pit)
        # num shape [S*A]
        num = np.einsum('sa,sa->sa', pr_pit, exp_eta_Q_pit)
        return num/(denom.reshape(-1, 1) + 1e-10)
    
    def get_policy_prob():
        pass


class DirectTabularACPGActor(Actor):
    def __init__(self, env, eta, init_theta, c):
        super().__init__(env, eta)

        # init_theta is |S*A| matrix with probs
        self.init_theta = copy.deepcopy(init_theta)

        # current theta
        self.current_theta = copy.deepcopy(init_theta)
        #self.current_theta = self.env.calc_pi_star()

        # c
        self.c = c

    def get_objective():
        pass
    
    def update_policy_param(self, Q_pit):
        pr_pit = copy.deepcopy(self.current_theta)
        self.current_theta = self.update_inner_weight(pr_pit, Q_pit)
        

    # The update is NPG. There is not inner loop update
    # p^{\pi}(a|s) = p^{\pi}_t (a|s) e^{\eta * Q^{\pi}_t(s, a)}} / \sum_b p^{\pi}_t(b|s) e^{\eta * Q^{\pi}_t(s, b)}}
    # etap = (eta * c) / (eta + c)
    def update_inner_weight(self, pr_pit, Q_pit):
        etap = (self.eta * self.c) / (self.eta + self.c)
        tmp = Q_pit - np.max(Q_pit, axis=1).reshape(-1, 1)
        exp_eta_Q_pit = np.exp(etap * tmp) 
        # denom shape [S]
        denom = np.einsum('sa,sa->s', pr_pit, exp_eta_Q_pit)
        # num shape [S*A]
        num = np.einsum('sa,sa->sa', pr_pit, exp_eta_Q_pit)
        return num/(denom.reshape(-1, 1) + 1e-10)
    
    def get_policy_prob():
        pass

class SoftmaxTabularsMDPOActor(Actor):
    def __init__(self, env, eta, init_theta):
        super().__init__(env, eta)

        # init_theta is |S*A| matrix with probs
        self.init_theta = copy.deepcopy(init_theta)

        # current theta
        self.current_theta = copy.deepcopy(init_theta)
    
    def get_objective():
        pass
    
    def update_policy_param(self, A_pit):
        pr_pit = copy.deepcopy(self.current_theta)
        self.current_theta = self.update_inner_weight(pr_pit, A_pit)
        

    # There is not inner loop update.
    # p^{\pi}(a|s) = p^{\pi}_t (a|s) max(1 + \eta * A^{\pi}_t(s, a), 0) / \sum_b p^{\pi}_t(b|s) max(1 + \eta * A^{\pi}_t(s, b), 0)}
    def update_inner_weight(self, pr_pit, A_pit):
        eps = 1e-8
        max_eta_A = np.max(1 + self.eta * A_pit, eps)
        # denom shape [S]
        denom = np.einsum('sa,sa->s', pr_pit, max_eta_A)
        # num shape [S*A]
        num = np.einsum('sa,sa->sa', pr_pit, max_eta_A)
        return num/(denom.reshape(-1, 1) + 1e-10)
    
    
    def get_policy_prob(self, theta):
        pass
    
    
class SoftmaxTabularACPGActor(Actor):
    def __init__(self, env, eta, init_theta, c):
        super().__init__(env, eta)

        # init_theta is |S*A| matrix with probs
        self.init_theta = copy.deepcopy(init_theta)

        # current theta
        self.current_theta = copy.deepcopy(init_theta)

        # c
        self.c = c
    
    def get_objective():
        pass
    
    def update_policy_param(self, A_pit):
        pr_pit = copy.deepcopy(self.current_theta)
        self.current_theta = self.update_inner_weight(pr_pit, A_pit)
        

    # There is not inner loop update.
    # p^{\pi}(a|s) = p^{\pi}_t (a|s) max(1 + \eta * A^{\pi}_t(s, a), 0) / \sum_b p^{\pi}_t(b|s) max(1 + \eta * A^{\pi}_t(s, b), 0)}
    def update_inner_weight(self, pr_pit, A_pit):
        eps = 1e-8
        etap = (self.eta * self.c) / (self.eta + self.c)
        max_eta_A = np.max(1 + etap * A_pit, eps)
        # denom shape [S]
        denom = np.einsum('sa,sa->s', pr_pit, max_eta_A)
        # num shape [S*A]
        num = np.einsum('sa,sa->sa', pr_pit, max_eta_A)
        return num/(denom.reshape(-1, 1) + 1e-10)
    
    
    def get_policy_prob(self, theta):
        pass