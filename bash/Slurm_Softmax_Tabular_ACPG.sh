#!/bin/bash


jobname='sof_acpg_tab'
filename="Softmax_Tabular_ACPG.sh"

for sampling in "MB" "MC"
do
    for eta in 0.001 0.01 0.1 1
    do
        for d in 40 60 80 100
        do
            sbatch -J $jobname $filename $eta $d $sampling
        done
    done
done