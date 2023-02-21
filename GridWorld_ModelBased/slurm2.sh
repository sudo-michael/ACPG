#!/bin/bash


for eta in 0.0001 0.001 0.01 0.1 1 10 100;
do
  for iht_size in 5 10 20 40 60 80;
  do
    for num_tiles in 1 2 3 4 5 10;
    do 
      for tiling_size in 1 2 3 4 5;
      do
        sbatch run2.sh $eta $iht_size $num_tiles $tiling_size
      done  
    done
  done
done