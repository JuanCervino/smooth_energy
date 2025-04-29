#!/bin/bash

n_trains=(2 4 6 8 10 12 14 16 18 20)  # Number of training samples

lip=250  
clamp=1000
dual_step=10

# n_trains=(20 22 24 26 28 30 32 34 36 38 40)  # Number of training samples
# lips=(80 70 60 50 40)  
# clamp=(1000 1000 1000 1000 1000)
# dual_step=(1 1 1 1 1)

for i in "${!n_trains[@]}"  # Loop over indices
do
    n_train="${n_trains[$i]}"
    n_unlab=$((20 - n_train))

    # lip="${lips[$i]}"
    # clamp="${clamp[$i]}"
    # dual_step="${dual_step[$i]}"
    echo "n_train=$n_train, lips=$lip", "clamp=$clamp, dual_step=$dual_step"
    # Call the sbatch command with the parameters
    sbatch run_single_gpu.sh "$n_train" "$lip" "$clamp" "$dual_step" "$n_unlab"


    # sbatch run_single_gpu.sh "$n_train" 

done
