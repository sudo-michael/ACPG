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

### Tabular Gridworld Environment

#### Model-based
Here, we assume that model (transition and reward) of environment is known.
* Folder **GridworldModelBased**

`Direct_TabularNPG_LFATD.py` and `Direct_TabularACPG_LFAACPG.py` contains the code for running TD algorithm, and our lagorithm for direct representation.
Similarlay, `Softmax_TabularsMDPO_LFATD.py` and `Softmax_TabularACPG_LFAACPG.py contains the code for softmax representation.


