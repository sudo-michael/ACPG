import numpy as np
import argparse
import os
from ACPG.Actor import DirectTabularACPGActor
from ACPG.Critic import LFAACPGCritic
from ACPG.environments.TabularMDPs import *
from ACPG.utils import *

def run(env, num_iterations, eta, init_theta, iht_size, num_tiles, tiling_size, c, lrc):
     
    # initialize actor object to update policy parameters
    actor = DirectTabularACPGActor(env, eta, init_theta, c)

    # initialize critic to estimate Q function. 
    #iht_size = env.state_space * env.action_space
    critic = LFAACPGCritic(env, iht_size, num_tiles, tiling_size, eta, c, lrc)

    # storing policies
    policy_list = []
    
    J_list= []
    J_grads = []
    
    # Storing Q function estimation error 
    diff_in_Q = []

    for t in range(num_iterations):
        
        print("Iteration:", t)
        print("#################")

        curr_policy = actor.current_theta
        policy_list.append(curr_policy)
        # set true state(action) value function (reward)


        Q = env.calc_qpi(curr_policy)
        V = env.calc_vpi(curr_policy)
        d = env.calc_dpi(curr_policy)

        J = np.dot(env.mu, V)
        J_list.append(J)
      


        # set estimated state(action) value function (reward)
        Q_hat = critic.get_estimated_Q(Q, curr_policy, d)

        # store Q_r estimation error
        diff_in_Q.append(np.linalg.norm(Q - Q_hat))

        # updating the primal policy
        actor.update_policy_param(Q_hat)
            


    return J_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=int, default=0)

    # number of iterations
    parser.add_argument('--num_iterations', help="iterations", type=int, default=150000)

    # Run is used to keep track of runs with different policy initialization.
    parser.add_argument('--run', help="run number", type=int, default=5)

    # 1/{\eta} is the parameter for divergence term
    parser.add_argument('--eta', help="divergence ", type=float, default=0.01)
    
    # hash table size
    parser.add_argument('--iht_size', help="iht size", type=int, default=60)

    # iht number of tiles
    parser.add_argument('--num_tiles', help="num of tiles", type=int, default=4)

    # size of tiling
    parser.add_argument('--tiling_size', help="dim of grid", type=int, default=3)

    # c
    parser.add_argument('--c', help="c", type=float, default=1)

    # critic inner loop
    #parser.add_argument('--mc', help="critic inner loop", type=int, default=1000)

    # critic lr
    parser.add_argument('--lrc', help="critic step size", type=float, default=1000)

    args = parser.parse_args()
    
    args.num_iterations = int(args.num_iterations)
    args.run = int(args.run)
    args.env = int(args.env)
    args.iht_size = int(args.iht_size)
    args.num_tiles = int(args.num_tiles)
    args.tiling_size = int(args.tiling_size)

    # creates a DST/CW based on arguments
    env = None
    if args.env == 0:
        env = get_CW()
    elif args.env == 1:
        env = get_DST()

    All_J_list = list()

    try:
        for i in range(args.run):
            rng = np.random.RandomState(i)
            init_theta = rng.dirichlet(np.ones(env.action_space), size=env.state_space)
            run_params = {'env': env,
                                'num_iterations': args.num_iterations,
                                'eta': args.eta,
                                'init_theta': init_theta,
                                'iht_size': args.iht_size,
                                'num_tiles': args.num_tiles,
                                'tiling_size': args.tiling_size,
                                'c': args.c,
                                'lrc': args.lrc
                                }
            
            J_list= run(**run_params)
            print(J_list)
            All_J_list.append(J_list)




        mean, interval = mean_confidence_interval(All_J_list)
        
        current_dir = os.path.join(os.path.abspath(os.getcwd()), "Results/DirectTabularACPGActor_LFAACPGCritic/")
        output_dir = dir_builder(args, current_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        np.save(output_dir+"/mean.npy", mean)
        np.save(output_dir+"/interval.npy", interval)

    except:
        print("error")

    

    

