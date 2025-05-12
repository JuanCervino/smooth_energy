#!/bin/bash
#SBATCH --job-name smooth_IL
#SBATCH -o ERM_smooth_IL_%j.log
#SBATCH --gres=gpu:volta:1

# Initialize the module command first source
source /etc/profile

# Then use the module command to load the module needed for your work
module load anaconda/Python-ML-2024b

nvcc --version
nvidia-smi

# Run experiment with the provided config
echo "Received config file:  $1 $2 $3 $4 $5 $6 $7 $8" 
python -m smooth.scripts.imitation_learning --epochs 50000 --bs 128 --lr 0.025  --n_train $1 --lipschitz_constant $2 --clamp_dual $3 --dual_step $4  --n_unlab $5 --seed $6 --dataset $7 --algorithm $8

# python -m smooth.scripts.imitation_learning --epochs 50000 --bs 128 --lr 0.025 --print_steps 1000 --algorithm ManifoldGradientBatch  --n_train $1 --lipschitz_constant $2 --clamp_dual $3 --dual_step $4 --n_unlab $5 --seed $6
