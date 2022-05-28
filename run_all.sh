#!/bin/bash

for dataset in adult avila_binary bank_full_binary card_clients covtype_binary egg_eye magic04 robot_nav_binary shuttle_binary irreducible_synthetic reducible_synthetic
do
  for sample_size in 800 
  do
    python3 run_experiments.py --dataset ${dataset} --sample_size ${sample_size} --relabel_frac 0.1 > ${dataset}_relabelf0.1_${sample_size}.out
  done
done
