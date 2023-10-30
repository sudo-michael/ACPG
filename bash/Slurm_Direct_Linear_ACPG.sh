#!/bin/bash


jobname='dir_acpg_lin'
filename="Direct_Linear_ACPG.sh"

for sampling in "MB" "MC"
do
    for eta in 0.005 0.01 0.1 1
    do
        for d in 40 60 80 100
        do
            sbatch -J $jobname $filename $eta $d $sampling
        done
    done
done