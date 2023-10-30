#!/bin/bash


jobname='dir_adv_tab'
filename="Direct_Tabular_AdvTD.sh"

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