#!/bin/bash


for eta in 0.0001 0.001 0.01 0.1 1 10 100;
do
  for epsilon in 0.1 0.2 0.4 0.6 0.8 1 2 3 4 5;
  do
    sbatch run1.sh $eta $epsilon
  done
done