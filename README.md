# ACPG

This repository contains code for running TD and ACPG algorithms in direct and softmax representations.

## Installation
* Create a virtual env using python3

`virtualenv -p python3 <envname>`

* Activate virtual environment

`source envname/bin/activate`

* Install other libraries using `requirements.txt` file

`pip install -r requirements.txt`

## How to run code?

### Tabular Gridorld Environment

#### Model-based
Here, we assume that model (transition and reward) of environment is known.
* Folder **GridWorld_ModelBased**

`Direct_TabularNPG_LFATD.py` and `Direct_TabularACPG_LFAACPG.py` contains the code for running TD algorithm, and our lagorithm for direct representation.
Similarlay, `Softmax_TabularsMDPO_LFATD.py` and `Softmax_TabularACPG_LFAACPG.py` contains the code for softmax representation.


Following shows the arguments for above scripts along with their default values. Note that some arguments are only for ACPG algorithm.
```
args = {
        "env": [0],
        "num_iterations":[150000],
        "run":[5],
        "eta":[0.01],
        "c":[0.01],
        "iht_size":[80],
        "num_tiles":[7],
        "tiling_size":[2],
        "mc": [10000], (only for ACPG)
        "lrc":[100000] (only for ACPG)
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

To run the TD algorithm for direct representation use:
`python GridWrold_ModelBased/Direct_TabularNPG_LFATD.py`

To run the ACPG algorithm for direct representation use:
`python GridWrold_ModelBased/Direct_TabularACPG_LFAACPG.py`

To run the TD algorithm for sofmtax representation use:
`python GridWrold_ModelBased/Softmax_TabularsMDPO_LFATD.py`

To run the ACPG algorithm for softmax representation use:
`python GridWrold_ModelBased/Softmax_TabularsMDPO_LFAACPG.py`


