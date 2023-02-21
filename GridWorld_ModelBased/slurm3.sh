#!/bin/bash


for eta in 0.001 0.01 0.1 1 10;
do
  for iht_size in 5 10 20 40 60 80;
  do
    for num_tiles in 1 3 5 10;
    do
      for tiling_size in 1 3;
      do 
        for c in 0.001 0.01 0.1 1 10;
        do
          for lrc in 100 1000;
          do
            sbatch run3.sh $eta $iht_size $num_tiles $tiling_size $c $lrc
          done
        done
      done
    done
  done
done