#!/bin/bash

n_trains=(2 4 6 8 10 12 14 16 18 20)

for config_file in "${n_trains[@]}"  # Loop through the list
do
    echo $config_file
    sbatch run_single_gpu.sh "$config_file"
done
