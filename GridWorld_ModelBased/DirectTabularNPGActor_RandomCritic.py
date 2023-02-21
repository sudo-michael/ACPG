import numpy as np
import argparse
import os
from ACPG.Actor import DirectTabularNPGActor
from ACPG.Critic import RandomCritic
from ACPG.environments.TabularMDPs import *
from ACPG.utils import *

def run(env, num_iterations, eta, init_theta, epsilon):

    # initialize actor object to update policy parameters
    actor = DirectTabularNPGActor(env, eta, init_theta)

    # initialize critic to estimate Q function. 
    critic = RandomCritic(env, epsilon)

    # storing policies
    policy_list = []
    
    #J = E[V(\pho)]
    J_list= []
    
    # Storing Q function estimation error 
    diff_in_Q= []

    for t in range(num_iterations):
        
        curr_policy = actor.current_theta
        policy_list.append(curr_policy)
        # set true state(action) value function (reward)

        Q = env.calc_qpi(curr_policy)
        V = env.calc_vpi(curr_policy)
        d = env.calc_dpi(curr_policy)
        
        J = np.dot(env.mu, V)
        J_list.append(J)

        # set estimated state(action) value function (reward)
        Q_hat = critic.get_estimated_Q(Q)

        # store Q_r estimation error
        diff_in_Q.append(np.linalg.norm(Q - Q_hat))
            
        # updating the primal policy
        actor.update_policy_param(Q_hat)

    return J_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=int, default=1)

    # number of iterations
    parser.add_argument('--num_iterations', help="iterations", type=int, default=1500)

    # Run is used to keep track of runs with different policy initialization.
    parser.add_argument('--run', help="run number", type=int, default=5)

    # 1/{\eta} is the parameter for divergence term
    parser.add_argument('--eta', help="divergence ", type=float, default=1)

    # step size to update omega 
    parser.add_argument('--epsilon', help="critic error ", type=float, default=0)


    args = parser.parse_args()
    
    args.num_iterations = int(args.num_iterations)
    args.run = int(args.run)
    args.env = int(args.env)
    

    # creates a DST/CW based on arguments
    env = None
    if args.env == 0:
        env = get_CW()
    elif args.env == 1:
        env = get_DST()

    All_J_list = list()
    for i in range(args.run):
        rng = np.random.RandomState(i)
        init_theta = rng.dirichlet(np.ones(env.action_space), size=env.state_space)
        run_params = {'env': env,
                            'num_iterations': args.num_iterations,
                            'eta': args.eta,
                            'init_theta': init_theta,
                            'epsilon': args.epsilon
                            }
        J_list= run(**run_params)
        All_J_list.append(J_list)
    
    mean, interval = mean_confidence_interval(All_J_list)
    
    current_dir = os.path.join(os.path.abspath(os.getcwd()), "Results/DirectTabularNPGActor_RandomCritic/")
    output_dir = dir_builder(args, current_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(output_dir+"/mean.npy", mean)
    np.save(output_dir+"/interval.npy", interval)



    

    

