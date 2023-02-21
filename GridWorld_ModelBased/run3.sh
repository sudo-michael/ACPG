#!/bin/bash
#SBATCH --mem=8192M	                      # Ask for 2 GB of RAM
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00                   # The job will run for 3 hours

eta=$1
iht_size=$2
num_tiles=$3
tiling_size=$4
c=$5
lrc=$6

file_name="DirectTabularACPGActor_LFAACPGCritic.py"

python $file_name --eta $eta --iht_size $iht_size --num_tiles $num_tiles --tiling_size $tiling_size --c $c --lrc $lrc