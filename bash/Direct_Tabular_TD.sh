#!/bin/bash
#SBATCH --ntasks-per-node=1          # Number of tasks per node
#SBATCH --cpus-per-task=16            # Number of CPUs per task
#SBATCH --mem=1G                     # Memory per node
#SBATCH --time=1-00:00:00              # Wall time limit

eta=$1
d=$2
sampling=$3

python -u main.py --env "CW" --sampling $sampling --critic_alg 'MSE' --representation 'direct' --actor_param 'tabular' --critic_d $d --eta $eta