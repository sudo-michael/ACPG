import numpy as np
import argparse
import os
import yaml
from Actor import DirectLinearMDPOActor, SoftmaxLinearsMDPOActor, DirectTabularNPGActor, SoftmaxTabularsMDPOActor
from Critic import DirectLFAACPGCritic, DirectLFATDCritic, SoftmaxLFATDCritic, SoftmaxLFAACPGCritic
from ACPG.environments.TabularMDPs import *
from ACPG.utils import *
from Sampling import MCSampling, TDSampling
import logging



def get_actor(cfg):

    actor = None
    ### get actor for linear paramterization in direct (linear mdpo) and softmax (linear smdpo) case
    if cfg['actor_param'] == 'linear':
        if cfg['representation'] == 'direct':
            actor = DirectLinearMDPOActor(
                cfg['env'],
                cfg['eta'], 
                cfg['actor_init_theta'], 
                cfg['actor_d'], 
                cfg['actor_num_tiles'], 
                cfg['actor_tiling_size'],
                cfg['actor_max_num_iterations'], 
                cfg['actor_max_lr'], 
                cfg['actor_stop_trs'],
                )
        elif cfg['representation'] == 'softmax':
            actor = SoftmaxLinearsMDPOActor(
                cfg['env'],
                cfg['eta'], 
                cfg['actor_init_theta'], 
                cfg['actor_d'], 
                cfg['actor_num_tiles'], 
                cfg['actor_tiling_size'],
                cfg['actor_max_num_iterations'], 
                cfg['actor_max_lr'], 
                cfg['actor_stop_trs'],
                )
    # get actor for tabular paramterization in direct (tabular npg/mdpo) and softmax (tabular smdpo) representations.
    elif cfg['actor_param'] == 'tabular':
        if cfg['representation'] == 'direct':
            actor = DirectTabularNPGActor(
                cfg['env'], 
                cfg['eta'], 
                cfg['actor_init_theta']
            )
        elif cfg['representation'] == 'softmax':
            actor = SoftmaxTabularsMDPOActor(
                cfg['env'], 
                cfg['eta'], 
                cfg['actor_init_theta']
            )
            print(f' actor {actor}')

    if cfg['c_in_stepsize']:
        actor.set_c(cfg['c'])
    
    return actor

def get_critic(cfg):

    critic = None

    ## get MSE (on Q) critic algorithm
    if cfg['critic_alg'] == 'MSE':
        critic = DirectLFATDCritic(
            cfg['env'], 
            cfg['critic_d'],
            cfg['critic_num_tiles'],
            cfg['critic_tiling_size']
        )
    
    # get AdvMSE (MSE on A) critic algorthm
    elif cfg['critic_alg'] == 'AdvMSE':
        critic = SoftmaxLFATDCritic(
            cfg['env'], 
            cfg['critic_d'],
            cfg['critic_num_tiles'],
            cfg['critic_tiling_size']
        )

    # get ACPG critic algorithm w.r.t. representation
    elif cfg['critic_alg'] == 'ACPG':
        if cfg['representation'] == 'direct':
            critic = DirectLFAACPGCritic(
                cfg['env'],
                cfg['critic_d'], 
                cfg['critic_num_tiles'], 
                cfg['critic_tiling_size'],
                cfg['eta'], 
                cfg['c'], 
                cfg['critic_max_num_iterations'],
                cfg['critic_max_lr'],
                cfg['critic_stop_trs'],
            )
        elif cfg['representation'] == 'softmax':
            critic = SoftmaxLFAACPGCritic(
                cfg['env'],
                cfg['critic_d'], 
                cfg['critic_num_tiles'], 
                cfg['critic_tiling_size'],
                cfg['eta'], 
                cfg['c'], 
                cfg['critic_max_num_iterations'],
                cfg['critic_max_lr'],
                cfg['critic_stop_trs'],
            )
    return critic



def get_sampler(cfg):
    sampler = None
    # get Monte Carlo sampler
    if cfg['sampling'] == "MC":
        sampler = MCSampling(cfg['env'], cfg['num_trajs'], cfg['episode_len'])
    
    return sampler

def run(cfg):
    # initialize actor/critic/sampler.
    actor = get_actor(cfg)

    critic = get_critic(cfg)

    sampler = get_sampler(cfg)

    # list to store results
    J_list= []
    
    
    for t in range(cfg['num_iterations']):
        
        ##### Be conservative in the beginning
        if t < 10:
            actor.eta = 0.01
        else:
            actor.eta = cfg['eta']
      

        # get the probabilities of the current policy
        curr_policy = actor.probs

        # get the true values from MDP
        Q = env.calc_qpi(curr_policy)
        V = env.calc_vpi(curr_policy)
        d = env.calc_dpi(curr_policy)

        # calculate value function and print it.
        J = np.dot(env.mu, V)
        if t % 1 == 0:
            logger.info(f'Iteration:{t}, J value: {J}')
        J_list.append(J)

        # Use sampler to obtain an estimate of Q (Q_sam)
        if sampler == None:
            Q_sam = Q
        elif isinstance(sampler, MCSampling):
            Q_sam = sampler.get_data(curr_policy)

        # get Q from the critic.
        Q_hat = critic.get_estimated_Q(Q_sam, curr_policy, d)


        # update the policy with Q_hat
        actor.update_policy_param(Q_hat)

    return J_list

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


    parser = argparse.ArgumentParser()

    # whether use configured params or new params.
    parser.add_argument('--load_config', help="load config", type=int, default=0)

    # whether run on Cliff World env or Frozen Lake env.
    parser.add_argument('--env', help="CW or FL", default="CW")

    # whether sample using Monte Carlo or Temporal difference or use known MDP. 
    parser.add_argument('--sampling', help="MB or MC or TD", default="MB")

    # maximum length of episode in the sampling. 
    parser.add_argument('--episode_len', help="episode length", default=20)

    # number of trajectories in sampling.
    parser.add_argument('--num_trajs', help='number of trajectories', default=1000)

    # number of outer iterations (policy updates).
    parser.add_argument('--num_iterations', help="iterations", type=int, default=40000)

    # number of runs. The initialized policy/ the initialized paramters in the critic and samples are different in each run.
    parser.add_argument('--run', help="run number", type=int, default=2)

    # 1/{\eta} is the parameter for divergence term
    parser.add_argument('--eta', help="divergence ", type=float, default=0.1)

    # whehther to use c in step size or not (only use $\eta$) in ACPG actor update
    parser.add_argument('--c_in_stepsize', help="0 or 1", type=int, default=0)

    ## paramter c in ACPG algorithm
    parser.add_argument('--c', help="c ", type=float, default=0.1)

    # Functional representation. Direct vs. Softmax
    parser.add_argument('--representation', help="functional representation", default="direct")

    # parameterization type of actor. Tabular vs. Linear
    parser.add_argument('--actor_param', help="linear or tabular", default="linear")

    # The capacity/dimension of the actor in linear paramterization
    parser.add_argument('--actor_d', help="actor linear dimension", default=80)

    # The number of tiles in actor featurization
    parser.add_argument('--actor_num_tiles', help="critic capacity", type=int, default=5)

    # The size of tiles in actor featurization
    parser.add_argument('--actor_tiling_size', help="critic capacity", type=int, default=3)

    # actor maximum number of inner loop/off-policy iterations
    parser.add_argument('--actor_max_num_iterations', help="actor maximum inner loop", type=int, default=10000) 

    # actor armijo maximum stepsize.
    parser.add_argument('--actor_max_lr', help="actor maximum step size", type=float, default=1000) 

    # actor threshold on grad norm to break the inner-loop. 
    parser.add_argument('--actor_stop_trs', help="actor stopping threshold", type=float, default=1e-4)

    # Type of critic algorithm
    parser.add_argument('--critic_alg', help="critic algorithm, MSE vs. AdvMSE vs. ACPG", default="MSE")

    # The capacity/dimension of the critic
    parser.add_argument('--critic_d', help="critic capacity", type=int, default=80)

    # The number of tiles in critic featurization
    parser.add_argument('--critic_num_tiles', help="critic capacity", type=int, default=5)

    # The size of tiles in critic featurization
    parser.add_argument('--critic_tiling_size', help="critic capacity", type=int, default=3)

    # critic maximum number of inner iterations
    parser.add_argument('--critic_max_num_iterations', help="critic maximum inner loop", type=int, default=10000) 

    # critic maximum armijo stepsize
    parser.add_argument('--critic_max_lr', help="critic maximum step size", type=float, default=1000)

    # critic threshold on grad norm to break the inner-loop. 
    parser.add_argument('--critic_stop_trs', help="critic stopping threshold", type=float, default=1e-6) 


    args = parser.parse_args()

    cfg = dict()
    # determine the environment.
    env = None
    if args.env == "CW":
        env = get_CW()
    elif args.env == "FL":
        env = get_FL()

    ###########################
    ## Construct config file ##
    ###########################

    cfg['env'] = env
    cfg['num_iterations'] = int(args.num_iterations)
    cfg['run'] = int(args.run)
    cfg['eta'] = args.eta

    # sampling params
    cfg['sampling'] = args.sampling
    cfg['episode_len'] = args.episode_len
    cfg['num_trajs'] = args.num_trajs

    # c params
    cfg['c'] = args.c
    cfg['c_in_stepsize'] = args.c_in_stepsize

    # functional representation
    cfg['representation'] = args.representation

    # actor params
    cfg['actor_param'] = args.actor_param
    cfg['actor_d'] = args.actor_d
    cfg['actor_max_lr'] = args.actor_max_lr
    cfg['actor_max_num_iterations'] = args.actor_max_num_iterations
    cfg['actor_stop_trs'] = args.actor_stop_trs
    cfg['actor_num_tiles'] = args.actor_num_tiles
    cfg['actor_tiling_size'] = args.actor_tiling_size

    # critic params
    cfg['critic_alg'] = args.critic_alg
    cfg['critic_d'] = args.critic_d
    cfg['critic_num_tiles'] = args.critic_num_tiles
    cfg['critic_tiling_size'] = args.critic_tiling_size
    cfg['critic_max_num_iterations'] = args.critic_max_num_iterations
    cfg['critic_max_lr'] = args.critic_max_lr
    cfg['critic_stop_trs'] = args.critic_stop_trs
    
    logger.info(f"Environment {args,env}")
    logger.info(f"Critic algorithm {cfg['critic_alg']}")
    logger.info(f"Functional representation {cfg['representation']}")
    logger.info(f"Actor parameterization {cfg['actor_param']}")
    logger.info(f"Sampling {cfg['sampling']}")
    logger.info("#######################")

    # lists to save results.
    All_J_list = list()

    # creating dir to save results
    current_dir = os.path.join(os.path.abspath(os.getcwd()), f'Results/{args.env}_{cfg["sampling"]}_{cfg["representation"]}_{cfg["critic_alg"]}_{cfg["actor_param"]}/')
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    output_dir = dir_builder(cfg, current_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # set seed
    seed = 42

    for i in range(cfg['run']):

        print("Run number: ", i)
        print("#######################")

        # setting the seed for run i
        rng = np.random.RandomState(seed * (i))
        
        # policy parameters initialization (w.r.t. paramterization type)
        if cfg['actor_param'] == 'linear':
            init_theta = rng.normal(0, 0.1, cfg['actor_d'])
        elif cfg['actor_param'] == 'tabular':
            init_theta = rng.dirichlet(np.ones(env.action_space), size=env.state_space)

        cfg['actor_init_theta'] = init_theta

        run_params = cfg

        ### run the algorithm
        J_list = run(run_params)
        All_J_list.append(J_list)

        # create dir for/save results of run i
        run_dir = output_dir + "/run" + str(i) + "/"
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        np.save(run_dir+"/J.npy", np.array(All_J_list[i]))

        print(f'Run {i} is finished')

    # get mean and interval of the exp.
    mean, interval = mean_confidence_interval(All_J_list)
    np.save(output_dir+"/mean.npy", mean)
    np.save(output_dir+"/interval.npy", interval)        
    print("Done!")


    

    

