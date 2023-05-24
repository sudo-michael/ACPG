# ACPG

This repository contains code of actor-critic algorithms in direct and softmax functional representations. The actor can be paramterized with tabular or linear paramterization.  The actor's policy update in direct case is MDPO algorithm, and in softmax case is sMDPO algorithm. The critic's paramterization is linear and its 
optimization can be TD,Advantage TD or decision-aware critic.

## Installation
* Create a virtual env using python3

`virtualenv -p python3 <envname>`

* Activate virtual environment

`source envname/bin/activate`

* Install other libraries using `requirements.txt` file

`pip install -r requirements.txt`

## How to run code?

### Tabular Gridorld Environment
We consider two gridworld environments, Cliff World and Frozen Lake, to test the actor-critic algorithms. All the files associated with gridworld environments are in `GridWorld` folder
In all experiments, the environment can be given as input to the running file using `--env` argument.
Also, whether the critic samples from the environment or use the known MDP as ground-truth can be set using `--sampling` argument.

#### Direct Representation / Tabular Actor

To run the TD algorithm with tabular actor using default hyper parameters use:

`python GridWrold/Direct_TabularNPG_LFATD.py --critic_alg TD`

To run the Advantage TD algorithm with tabular actor using default hyper parameters use:

`python GridWrold/Direct_TabularNPG_LFATD.py --critic_alg AdvTD`

To run the decision-aware algorithm with tabular actor using default hyper parameters use:

`python GridWrold/Direct_TabularACPG_LFAACPG.py`


#### Direct Representation / Linear Actor

To run the TD algorithm with linear actor using default hyper parameters use:
`python GridWrold/Direct_LinearMDPO_LFATD.py --critic_alg TD`

To run the Advantage TD algorithm with linear actor using default hyper parameters use:
`python GridWrold/Direct_LinearMDPO_LFATD.py --critic_alg AdvTD`

To run the decision-aware algorithm with linear actor using default hyper parameters use:
`python GridWrold/Direct_LinearACPG_LFAACPG.py`





Following shows the arguments for above scripts along with their default values. You can modify hyper parameter values by specifying their name and their values. Also, note that some arguments are only for ACPG algorithm.

```
args = {
        "env": [0],
        "num_iterations":[150000],
        "run":[5],
        "eta":[0.01],
}
"env": env=0 is Cliff world, env=1 is Deep sea treasure
"num_iterations": number of actor iterations for update
"run": number of runs
"eta": actor step size
"c": 
"iht_size":  feature dimension 
"num_tiles": number of tiles in tile coding feature generation
"tiling_size": size of tiles in tile coding feature generation
"mc": maximum number of iterations for critic optimization
"lrc": maximum step size for critic optimization
```
