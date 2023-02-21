#!/bin/bash
#SBATCH --mem=1024M	                      # Ask for 2 GB of RAM
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00                   # The job will run for 3 hours

eta=$1
iht_size=$2
num_tiles=$3
tiling_size=$4

file_name="DirectTabularNPGActor_LFATDCritic.py"

python $file_name --eta $eta --iht_size $iht_size --num_tiles $num_tiles --tiling_size $tiling_size