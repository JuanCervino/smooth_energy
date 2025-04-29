#!/bin/bash
#SBATCH --job-name imitation_smoothness
#SBATCH -o energy_imitation_%j.log
#SBATCH --gres=gpu:volta:1

# Initialize the module command first source
source /etc/profile

# Then use the module command to load the module needed for your work
module load anaconda/Python-ML-2024b
nvcc --version
nvidia-smi

python --version

python -m smooth.scripts.dataset_lipschitz_calculation