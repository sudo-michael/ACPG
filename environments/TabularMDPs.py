import numpy as np
import copy
import gym
from gym import utils

#### The script is for creating new Tabular MDPs. CliffWorld and FrozenLake are two 
#### examples implemented here using Tabular MDP class and we can extend it to other MDPs.


class TabularMDP:
    def __init__(self, P, r, mu, terminal_states, gamma, episode_cutoff_length,
                 reward_noise):
        """
        Parameter:
        P: [S * A * S] - transition prbability matrix 
        r: [S * A] - reward matrix
        mu: [S] - initial state distribution
        terminal_states: list of terminal states
        gamma: float - discount factor
        episode_cutoff_length: maximum lenght of an episode
        reward_noise: noise on reward function
        """
        self.P = P
        self.r = r
        self.mu = mu
        self.terminal_states = terminal_states
        
        self.state_space = self.r.shape[0]
        self.action_space = self.r.shape[1]

        self.gamma = gamma
        self.episode_cutoff_length = episode_cutoff_length
        self.reward_noise = reward_noise

        # lenght of episode up to now
        self.t = None

        # current state in episoe
        self.state = None

    # reset an episode, set t = 0 and initialiaze first state
    def reset(self):
        self.t = 0
        self.state = np.random.choice(a=self.state_space, p=self.mu)
        return self.state

    # takes an action and has a current state in self.state -> outputs reward, next state, done
    def step(self, action):
        if self.state is None:
            raise Exception('step() used before calling reset()')
        assert action in range(self.action_space)

        reward = self.r[self.state, action]+ np.random.normal(loc=0, scale=self.reward_noise)
        self.state = np.random.choice(a=self.state_space, p=self.P[self.state, action])
        self.t = self.t + 1

        done = 'false'
        if self.state in self.terminal_states:
            done = 'terminal'
        elif self.t > self.episode_cutoff_length:
            done = 'cutoff'

        return self.state, reward, done

    # for computing v_pi = (I - gamma P_pi)^{-1} r_pi
    def calc_vpi(self, pi, FLAG_RETURN_V_S0=False):
        # pi : [S*A] policy
        # p_pi : [S*S] -  P_pi(s' | s) = sum_a P(s' | s, a) * pi(a | s)
        p_pi = np.einsum('xay,xa->xy', self.P, pi) 

        #r_pi :[S] r_pi(s) = sum_a r(s, a) pi(a|s)
        r_pi = np.einsum('xa,xa->x', self.r, pi)

        # v_pi: [S] value function of states under pi
        v_pi = np.linalg.solve(np.eye(self.state_space) - self.gamma * p_pi, r_pi)

        
        if FLAG_RETURN_V_S0: # calculate v_s0
            v_s0 = np.dot(self.mu, v_pi)
            return v_s0
        else:
            return v_pi

    # for computing q_pi = r(s, a) + gamma * sum_s' p(s' | s, a) * v^pi(s')
    def calc_qpi(self, pi):
        v_pi = self.calc_vpi(pi)

        # q_pi:[S*A] state-action value function under pi
        q_pi = self.r + self.gamma * np.einsum('xay,y->xa', self.P, v_pi)
        return q_pi

    # for computing the normalized occupancy measure
    # d_pi = (1 - gamma) * mu (I - gamma P_pi)^{-1};   mu = initial state distribution
    def calc_dpi(self, pi):
        # p_pi : [S*S]  p_pi(s' | s) = sum_a P(s' | s, a) * pi(a | s)
        p_pi = np.einsum('xay,xa->xy', self.P, pi)
        d_pi = (1 - self.gamma) * np.linalg.solve(np.eye(self.state_space) - self.gamma * p_pi.T, self.mu)
        d_pi /= d_pi.sum() # for addressing numerical errors
        return d_pi
    
    
    # for computing state action value function under optimal policy
    def calc_q_star(self, num_iters=1000):
        q = np.zeros((self.state_space, self.action_space))
        for i in range(num_iters):
            q_new = self.r + np.einsum("xay,y->xa", self.gamma * self.P, q.max(1))
            q = q_new.copy()
        return q


    # for computing value function under optimal policy
    def calc_v_star(self, num_iters=1000):
        v = np.zeros(self.state_space)
        for i in range(num_iters):
            v_new = self.r + np.einsum("xay,y->xa", self.gamma * self.P, v)
            v_new = v_new.max(1)
            v = v_new.copy()
        return v

    # for computing optimal policy
    def calc_pi_star(self, num_iters=1000): # just go greedy wrt q_star
        q_star = self.calc_q_star(num_iters=num_iters)
        pi_star = np.zeros((self.state_space, self.action_space))
        pi_star[range(self.state_space), q_star.argmax(1)] = 1
        return pi_star


class TabularMDPstc:
    def __init__(self, P, r, mu, terminal_states, gamma, episode_cutoff_length, reward_noise):
        """
        Parameter:
        P: [S * A * S] - transition prbability matrix 
        r: [S * A * S] - reward matrix
        mu: [S] - initial state distribution
        terminal_states: list of terminal states
        gamma: float - discount factor
        episode_cutoff_length: maximum lenght of an episode
        reward_noise: noise on reward function
        """
        self.P = P
        print(self.P[13, 1, :])
        self.r = r
        self.mu = mu
        self.terminal_states = terminal_states
        self.state_space = self.r.shape[0]
        self.action_space = self.r.shape[1]
        self.gamma = gamma
        self.episode_cutoff_length = episode_cutoff_length
        self.reward_noise = reward_noise
        self.t = None
        self.state = None
    
    def reset(self):
        self.t = 0
        self.state = np.random.choice(a=self.state_space, p=self.mu)
        return self.state

    def step(self, action):
        if self.state is None:
            raise Exception('step() used before calling reset()')
        assert action in range(self.action_space)
        
        next_state = np.random.choice(a=self.state_space, p=self.P[self.state, action])
        reward = self.r[self.state, action, next_state] + np.random.normal(loc=0, scale=self.reward_noise)
        
        self.t = self.t + 1
        self.state = next_state

        done = False
        if self.state in self.terminal_states:
            done = True
        elif self.t > self.episode_cutoff_length:
            done = True

        return self.state, reward, done

    def calc_vpi(self, pi, FLAG_RETURN_V_S0=False):
        p_pi = np.einsum('xay,xa->xy', self.P, pi)
        r_det = np.einsum('xay,xay->xa', self.r, self.P) 
        r_pi = np.einsum('xa,xa->x', r_det, pi)
        v_pi = np.linalg.solve(np.eye(self.state_space) - self.gamma * p_pi, r_pi)

        if FLAG_RETURN_V_S0:
            v_s0 = np.dot(self.mu, v_pi)
            return v_s0
        else:
            return v_pi

    def calc_qpi(self, pi):
        v_pi = self.calc_vpi(pi)
        q_pi = np.zeros((self.state_space, self.action_space))
        for s in range(self.state_space):
            for a in range(self.action_space):
                q_pi[s,a] = np.sum(self.P[s,a,:] * (self.r[s,a,:] + self.gamma * v_pi))
        return q_pi

    def calc_dpi(self, pi):
        p_pi = np.einsum('xay,xa->xy', self.P, pi)
        d_pi = (1 - self.gamma) * np.linalg.solve(np.eye(self.state_space) - self.gamma * p_pi.T, self.mu)
        d_pi /= d_pi.sum() # for addressing numerical errors
        return d_pi

    # for computing state action value function under optimal policy
    def calc_q_star(self, num_iters=1000):
        q = np.zeros((self.state_space, self.action_space))
        for i in range(num_iters):
            q_new = np.einsum('say,say->sa', self.P, self.r) + np.einsum("xay,y->xa", self.gamma * self.P, q.max(1))
            q = q_new.copy()
        return q


    # for computing value function under optimal policy
    def calc_v_star(self, num_iters=1000):
        v = np.zeros(self.state_space)
        for i in range(num_iters):
            v_new = np.einsum('say,say->sa', self.P, self.r) + np.einsum("say,y->sa", self.gamma * self.P, v)
            v_new = v_new.max(1)
            v = v_new.copy()
        return v

    # for computing optimal policy
    def calc_pi_star(self, num_iters=1000): # just go greedy wrt q_star
        q_star = self.calc_q_star(num_iters=num_iters)
        pi_star = np.zeros((self.state_space, self.action_space))
        pi_star[range(self.state_space), q_star.argmax(1)] = 1
        return pi_star


def get_CW():
    terminal_states = [20]
    P = np.zeros((21, 4, 21))
    for state_idx in range(21):
        for action_idx in range(4):
            if state_idx in [1, 2, 3]: # chasms: reset to start state 0
                new_state_idx = 0
            elif state_idx == 4: # goal state: agent always goes to 20
                new_state_idx = 20
            elif state_idx == 20: # terminal state
                new_state_idx = 20
            else: # move according to the deterministic dynamics
                x_new = x_old = state_idx // 5
                y_new = y_old = state_idx % 5
                if action_idx == 0: # Down
                    y_new = np.clip(y_old - 1, 0, 4)
                elif action_idx == 1: # Up
                    y_new = np.clip(y_old + 1, 0, 4)
                elif action_idx == 2: # Left
                    x_new = np.clip(x_old - 1, 0, 3)
                elif action_idx == 3: # Right
                    x_new = np.clip(x_old + 1, 0, 3)
                new_state_idx = 5 * x_new + y_new

            P[state_idx, action_idx, new_state_idx] = 1
            
    r = np.zeros((21, 4))
    r[1, :] = r[2, :] = r[3, :] = -100 # negative reward for falling into chasms
    r[4, :] = +1 # positive reward for finding the goal terminal state

    
    mu = np.zeros(21)
    mu[0] = 1

    P_CliffWorld = copy.deepcopy(P)
    r_CliffWorld = copy.deepcopy(r)
    mu_CliffWorld = copy.deepcopy(mu)
    terminal_states_CliffWorld = copy.deepcopy(terminal_states)

    env_CW = TabularMDP(
    P=P_CliffWorld, r=r_CliffWorld, mu=mu_CliffWorld,
    terminal_states=terminal_states_CliffWorld,
    gamma=0.9, episode_cutoff_length=1000, reward_noise=0)

    return env_CW

def get_FL(gamma=0.9):
    env = gym.make('FrozenLake-v1')
    env_dict = env.P
    n_states = 16
    n_actions = 4

    # Initialize the reward and transition probability matrices
    reward = np.zeros((n_states, n_actions, n_states))
    P = np.zeros((n_states, n_actions, n_states))
    terminal_states = []

    # Convert the dictionary to NumPy arrays
    for s in range(n_states):
        for a in range(n_actions):
            for i in range(len(env_dict[s][a])):
                next_s, r, done = env_dict[s][a][i][1], env_dict[s][a][i][2], env_dict[s][a][i][3]
                reward[s, a, next_s] = r
                P[s, a, next_s] += env_dict[s][a][i][0]
                if done == True:
                    terminal_states.append(next_s)
    
    terminal_states = list(set(terminal_states))
    mu = np.zeros(n_states)
    mu[0] = 1

    P_FrozenLake = copy.deepcopy(P)
    r_FrozenLake = copy.deepcopy(reward)
    mu_FrozenLake = copy.deepcopy(mu)
    terminal_states_FrozenLake = copy.deepcopy(terminal_states)

    env_FL = TabularMDPstc(
    P=P_FrozenLake, r=r_FrozenLake, mu=mu_FrozenLake,
    terminal_states=terminal_states_FrozenLake,
    gamma=gamma, episode_cutoff_length=1000, reward_noise=0)

    return env_FL

 
 
 





