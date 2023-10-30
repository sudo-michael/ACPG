#!/bin/bash

jobname='soft_td_lin'
filename="Softmax_Linear_TD.sh"

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