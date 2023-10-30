# ACPG

This repository contains implementations of actor-critic algorithms described in 'Decision-Aware Actor-Critic with Function Approximation and Theoretical Guarantees' paper (https://arxiv.org/abs/2305.15249) on two grid-world environments. For running the algorithms please install the required packages and run the following commands. 

![Comparison of three critic objective functions with varying capacity](figs/.png)

## Installation
* Create a virtual env using python3.
`virtualenv -p python3 <envname>`

* Activate the virtual environment.
`source envname/bin/activate`

* Clone the repo, and install other python packages with `requirements.txt` file.
`pip install -r requirements.txt`

## Running the code
In order to execute the algorithms, you should run `main.py` with the corresponding arguments. 
* You can use methods on two gridworld environments, Cliff World (CW) and Frozen Lake (FL), which be given as input with `--env` argument. 

* We considered three critic algorithms, MSE (squared loss functions for `Q`), AdvMSE (squared loss functions for `A`) and ACPG (decision-aware loss functions) which can be given as input with `--critic_alg` argument.

* We considered two funational representations, direct and softmax, which can be given as input with `--representation` argument.

* We considered two paramterization for actor policy, linear or tabular, which can be given as input with `--actor_param` argument.

* You can also choose to use the true MDP or sample using Monte Carlo which can be determined with `sampling` argument.

For instance, the following runs ACPG method on Cliff World environment with a linear actor (d=80) on direct representation, and the agent samples using Monte Carlo.
```
python -u main.py --env "CW" --sampling MC --critic_alg 'ACPG' --representation 'direct' --actor_param 'linear' --critic_d 80
```

