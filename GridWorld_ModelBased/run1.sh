#!/bin/bash
#SBATCH --mem=1024M	                      # Ask for 2 GB of RAM
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00                   # The job will run for 3 hours

eta=$1
epsilon=$2

file_name="DirectTabularNPGActor_RandomCritic.py"

python $file_name --eta $eta --epsilon $epsilon