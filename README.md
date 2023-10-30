# ACPG

This repository contains implementations of actor-critic algorithms described in 'Decision-Aware Actor-Critic with Function Approximation and Theoretical Guarantees' paper on two grid-world environments. You can find the paper here: arxiv.org/abs/2305.15249. For running the algorithms please install the required packages and run the following commands. 

## Algorithms.
We compare the proposed decision-aware critic loss function with squared (TD) loss functions (either on value function Q or advantage function A) when coupled with a linearly parameterized actor. The actor can view/update the policy in direct or softamx functional representations. 

## Installation
* Create a virtual env using python3.

`virtualenv -p python3 <envname>`

* Activate the virtual environment.

`source envname/bin/activate`

* Clone the repo, and install other python packages with `requirements.txt` file.

`pip install -r requirements.txt`



### Tabular Gridorld Environment
We consider two gridworld environments, Cliff World and Frozen Lake, to test the actor-critic algorithms. All the files associated with gridworld environments are in `GridWorld` folder
In all experiments, the environment can be given as input to the running file using `--env` argument.
Also, whether the critic samples from the environment or use the known MDP as ground-truth can be set using `--sampling` argument.

#### Direct Representation / Tabular Actor

To run the TD algorithm with default parameters use:

`python GridWrold/Direct_TabularNPG_LFATD.py --critic_alg TD`

To run the Advantage TD algorithm with default parameters use:

`python GridWrold/Direct_TabularNPG_LFATD.py --critic_alg AdvTD`

To run the decision-aware algorithm with default parameters use:

`python GridWrold/Direct_TabularACPG_LFAACPG.py`


#### Direct Representation / Linear Actor

To run the TD algorithm with default parameters use:

`python GridWrold/Direct_LinearMDPO_LFATD.py --critic_alg TD`

To run the Advantage TD algorithm with default parameters use:

`python GridWrold/Direct_LinearMDPO_LFATD.py --critic_alg AdvTD`

To run the decision-aware algorithm with default parameters use:

`python GridWrold/Direct_LinearACPG_LFAACPG.py`

#### Softmax Representation / Tabular Actor

To run the TD algorithm with default parameters use:

`python GridWrold/Softmax_TabularsMDPO_LFATD.py --critic_alg TD`

To run the Advantage TD algorithm with default parameters use:

`python GridWrold/Softmax_TabularsMDPO_LFATD.py --critic_alg AdvTD`

To run the decision-aware algorithm with default parameters use:

`python GridWrold/Softmax_TabularACPG_LFAACPG.py`


#### Softmax Representation / Linear Actor

To run the TD algorithm with default parameters use:

`python GridWrold/Softmax_LinearsMDPO_LFATD.py --critic_alg TD`

To run the Advantage TD algorithm with default parameters use:

`python GridWrold/Softmax_LinearsMDPO_LFATD.py --critic_alg AdvTD`

To run the decision-aware algorithm with default parameters use:

`python GridWrold/Softmax_LinearACPG_LFAACPG.py`

Following shows the arguments for the script `Softmax_LinearACPG_LFAACPG.py` along with their default values. You can modify hyper parameter values by specifying their name and their values. Also, note that some arguments are only for ACPG algorithm and some arguments are only for the case that actor in parameterized linearly.

```
args = {
        "env": ["CW"],
        "sampling": ["MB"],
        "num_iterations":[50000],
        "run":[5],
        "eta":[0.01],
        "d": [80],
        "actor_stop_trs": [1e-4],
        "actor_max_num_iterations": [1000],
        "actor_max_lr" : [1000],
        "critic_stop_trs: [1e-6],
        "critic_max_num_iterations": [1000],
        "critic_max_lr" : [1000],
        "init_c": [0.01],
        "c_in_stepsize": [0]
        
}
"env": env=CW is Cliff world, env=FL is Frozen Lake
"sampling": sampling=MB is known MDP, sampling=MC is Monte Carlo sampling
"num_iterations": number of actor policy updates
"run": number of distinct runs
"eta": actor functional step size
"d": critic dimension/expressivity
"actor_stop_trs": threshold on gradient norm to stop inner-loop (Only for linear actor)
"actor_max_num_iterations": maximum number of iteration in inner-loop (Only for linear actor)
"actor_max_lr" : Armijo maximum step size  (Only for linear actor)
"critic_stop_trs": threshold on gradient norm to stop inner-loop (Only for decision-aware)
"critic_max_num_iterations": maximum number of iteration in inner-loop (Only for decision-aware)
"critic_max_lr" : Armijo maximum step size  (Only for decision-aware)
"init_c": trade-off parameter (Only for decision-aware)
"c_in_stepsize": whether to involve `c` in functional step size or not (Only for decision-aware)
```
