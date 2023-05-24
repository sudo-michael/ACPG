import numpy as np
import argparse
import os
import time
import yaml
import math
from ACPG.Actor import SoftmaxTabularsMDPOActor
from ACPG.Critic import SoftmaxLFAACPGCritic
from ACPG.environments.TabularMDPs import *
from ACPG.utils import *
from Sampling import MCSampling
def run(env, num_iterations, eta, d, num_tiles, tiling_size, \
    c, c_in_stepsize, critic_m, critic_lr, critic_stop_trs, actor_init_theta, sampling):
    # initialize actor object to update policy parameters
    actor = SoftmaxTabularsMDPOActor(env, eta, c_in_stepsize, actor_init_theta)
    if c_in_stepsize:
        actor.set_c(c)

    critic = SoftmaxLFAACPGCritic(env, d, num_tiles, tiling_size, eta, c, critic_m, critic_lr, critic_stop_trs)

    # use samples or not.
    sampler = None
    if sampling == "MC":
        sampler = MCSampling(env, 1000, 20)

    # lists to store results
    policy_list = []
    J_list= []
    critic_loss_list = []
    critic_grad_list = []
    actor_grad_list = []
    

    for t in range(num_iterations):

        # in the first 20 iterations be more conservative
        if t < 20:
            actor.eta = 0.01
        else:
            actor.eta = eta
        
        # get/save the probabilities of the current policy
        curr_policy = actor.get_policy_prob(actor.current_theta)
        policy_list.append(curr_policy)

        # get the true values from MDP
        Q = env.calc_qpi(curr_policy)
        V = env.calc_vpi(curr_policy)
        A = Q - V.reshape(-1, 1)
        d = env.calc_dpi(curr_policy)

        # calc/save value function and print it every 100 its.
        J = np.dot(env.mu, V)
        if t % 1 == 0:
            print("#################")
            print("Iteration:", t)
            print("J", J)
        J_list.append(J)

        # get/save policy grad norm
        actor_grad = actor.get_grad(A, curr_policy, d)
        actor_grad_list.append(np.linalg.norm(actor_grad))

        # convert Q to Q_sam
        if sampler == None:
            Q_sam = Q
        elif isinstance(sampler, MCSampling):
            Q_sam = sampler.get_data(curr_policy)

        A_sam = Q_sam - np.einsum('sa,sa->s', curr_policy, Q_sam).reshape(-1, 1)    
        A_hat = critic.get_estimated_A(A_sam, curr_policy, d)

        # get/save the critic loss/grad
        critic_loss = critic.get_loss(A, curr_policy, d, critic.theta)
        critic_loss_list.append(critic_loss)

        critic_grad = critic.get_gradient(A, curr_policy, d, critic.theta)
        critic_grad_list.append(np.linalg.norm(critic_grad))

        # update the policy with Q_hat
        actor.update_policy_param(A_hat)


    return J_list, critic_loss_list, critic_grad_list, actor_grad_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # whether use configured params or new params.
    parser.add_argument('--is_config', help="load config", type=int, default=0)

    # whether sample using MC or use known MDP. 
    parser.add_argument('--sampling', help="MB or MC", default="MB")
    
    # whether run on CW env or FL env.
    parser.add_argument('--env', help="CW or FL", default="CW")

    # number of iterations (policy updates)
    parser.add_argument('--num_iterations', help="iterations", type=int, default=50000)

    # number of runs. The initialized policy and/or MC samples are different in each run.
    parser.add_argument('--run', help="run number", type=int, default=5)

    # 1/{\eta} is the parameter for divergence term
    parser.add_argument('--eta', help="divergence ", type=float, default=0.1)
    
    # The capacity/dimension of the critic
    parser.add_argument('--d', help="critic capacity", type=int, default=80)
    
    # critic maximum number of iteration (per policy update)
    parser.add_argument('--critic_max_num_iterations', help="critic maximum inner loop", type=int, default=10000) 

    # critic maximum armijo stepsize
    parser.add_argument('--critic_max_lr', help="critic maximum step size", type=float, default=1000)

    # actor threshold on grad norm to break the inner-loop. 
    parser.add_argument('--critic_stop_trs', help="critic stopping threshold", type=float, default=1e-6) 

    # trade-off paramter c
    parser.add_argument('--init_c', help="c", type=float, default=1e-2)

    # whehther to use c in step size or not (only use $\eta$)
    parser.add_argument('--c_in_stepsize', help="0 or 1", type=int, default=0)


    args = parser.parse_args()
    cfg = None

    # determine the environment.
    env = None
    if args.env == "CW":
        env = get_CW()
    elif args.env == "FL":
        env = get_FL()
    
    #determine the critic dimension. 
    args.d = int(args.d)

    # use config file
    if args.is_config == 1:
        try:
            # directory of config file : env + sampling + representaion + alg + critic dimenstion
            yaml_file_directory = 'Configs/' + args.env + "_" + args.sampling + "_Softmax_TabularACPG_" + str(args.d) + ".yaml"
            with open(yaml_file_directory) as file:
                parameter_dic = yaml.load(file, Loader=yaml.FullLoader)
            cfg = parameter_dic['parameters']
        except:
            print("Config file has not been created")
    
    # use new params
    elif args.is_config == 0:
        cfg = dict()

        # add new params to cfg 
        cfg['num_iterations'] = int(args.num_iterations)
        cfg['run'] = int(args.run)
        cfg['eta'] = args.eta

        # c params
        cfg['init_c'] = args.init_c
        cfg['c_in_stepsize'] = args.c_in_stepsize

        # critic params
        cfg['critic_max_num_iterations'] = args.critic_max_num_iterations
        cfg['critic_max_lr'] = args.critic_max_lr
        cfg['critic_stop_trs'] = args.critic_stop_trs
    
    # add env, sampling and critic dim to cfg
    cfg['d'] = args.d
    cfg['sampling'] = args.sampling
    cfg['env'] = args.env


    # featurization (d + num_tiles + tiling_size)
    if args.env =="CW":
        # critic featurization based on given dimension
        if args.d == 40:
            cfg['num_tiles'] = 5
            cfg['tiling_size'] = 1

        elif args.d == 50:
            cfg['num_tiles'] = 4
            cfg['tiling_size'] = 2

        elif args.d == 60:
            cfg['num_tiles'] = 4
            cfg['tiling_size'] = 3

        elif args.d == 80:
            cfg['num_tiles'] = 5
            cfg['tiling_size'] = 3

        elif args.d == 100:
            cfg['num_tiles'] = 6
            cfg['tiling_size'] = 3

    elif args.env == "FL":  

        # critic featurization based on given dimension
        if args.d == 40: 
            cfg['num_tiles'] = 3
            cfg['tiling_size'] = 3

        elif args.d == 50: 
            cfg['num_tiles'] = 4
            cfg['tiling_size'] = 3

        elif args.d == 60: 
            cfg['num_tiles'] = 5
            cfg['tiling_size'] = 3

        elif args.d == 100: 
            cfg['num_tiles'] = 8
            cfg['tiling_size'] = 3

    print(cfg)

    # lists to save results.
    All_J_list = list()
    All_c_list = list()
    All_critic_loss_list = list()
    All_critic_grad_list = list()
    All_actor_grad_list = list()

    # creating dir to save results
    current_dir = os.path.join(os.path.abspath(os.getcwd()), "Results/Softmax_TabularACPG_LFAACPG/"+ cfg['env'] + "_" +cfg['sampling'] + "/")
    output_dir = dir_builder(cfg, current_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    # set seed.
    seed = 42
    for i in range(cfg['run']):
        print("Run number: ", i)
        print("#################")

        # setting the seed for run i
        rng = np.random.RandomState(seed * (i))
        
        # policy parameters initialization 
        init_theta = rng.dirichlet(np.ones(env.action_space), size=env.state_space)

        # constructing params of actor-critic alg.
        run_params = {'env': env,
                            'num_iterations': cfg['num_iterations'],
                            'eta': cfg['eta'],

                            'd': cfg['d'],
                            'num_tiles': cfg['num_tiles'],
                            'tiling_size': cfg['tiling_size'],

                            'c': cfg['init_c'],
                            'c_in_stepsize' : cfg['c_in_stepsize'],

                            'critic_m': cfg['critic_max_num_iterations'],
                            'critic_lr': cfg['critic_max_lr'],
                            'critic_stop_trs':cfg['critic_stop_trs'],

                            'actor_init_theta': init_theta,

                            'sampling': cfg['sampling']
                            }
        # get/save results of run i
        J_list, critic_loss_list, critic_grad_list, actor_grad_list = run(**run_params)
        All_J_list.append(J_list)
        All_critic_loss_list.append(critic_loss_list)
        All_critic_grad_list.append(critic_grad_list)
        All_actor_grad_list.append(actor_grad_list)

        # create dir for/save results of run i
        run_dir = output_dir + "/run" + str(i) + "/"
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        np.save(run_dir+"/J.npy", np.array(All_J_list[i]))
        np.save(run_dir+"/critic_loss.npy", np.array(All_critic_loss_list[i]))
        np.save(run_dir+"/critic_grad.npy", np.array(All_critic_grad_list[i]))
        np.save(run_dir+"/actor_grad.npy", np.array(All_actor_grad_list[i]))

    # get mean, interval of the exp.
    mean, interval = mean_confidence_interval(All_J_list)
    np.save(output_dir+"/mean.npy", mean)
    np.save(output_dir+"/interval.npy", interval)        
    print("Done!")
    

    

