#!/bin/bash

# n_trains=(2 4 6 8 10 12 14 16 18 20)  # Number of training samples
# seeds=(1 2 3)



datasets=("swimmer" "ant" "halfcheetah" "walker2d")
# datasets=("halfcheetah")

n_trains=(2 4 6 8 10 12 14 16 18 20)  # Number of training samples
seeds=(10)
lip=250  
clamp=1000
dual_step=10
algorithm="ERM"  # Algorithm to use


for dataset in "${datasets[@]}"
do
    for seed in "${seeds[@]}"
    do
        # echo "
        for i in "${!n_trains[@]}"  # Loop over indices
        do
            n_train="${n_trains[$i]}"
            n_unlab=$((20 - n_train))

            # lip="${lips[$i]}"
            # clamp="${clamp[$i]}"
            # dual_step="${dual_step[$i]}"
            echo "Running experiments n_train=$n_train, lips=$lip", "clamp=$clamp, dual_step=$dual_step", "n_unlab=$n_unlab, seed=$seed, dataset=$dataset", "algorithm=$algorithm"
            # Call the sbatch command with the parameters
            sbatch run_single_gpu.sh "$n_train" "$lip" "$clamp" "$dual_step" "$n_unlab" "$seed" "$dataset" "$algorithm"
        done
    done
done