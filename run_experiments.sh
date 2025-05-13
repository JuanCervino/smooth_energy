#!/bin/bash
#SBATCH --job-name imitation_smoothness
#SBATCH -o energy_imitation_%j.log
#SBATCH --gres=gpu:volta:1

# Initialize the module command first source
source /etc/profile

# Then use the module command to load the module needed for your work
module load anaconda/Python-ML-2024b
# module load anaconda/Python-ML-2023b # clu issue

nvcc --version
nvidia-smi


python -m smooth.scripts.imitation_learning --epochs 70000 --n_train 2 --n_test 2 --bs 128 --lr 0.025 --heat_kernel_t 0.1 --clamp 0.09  --algorithm ManifoldGradientBatch --regularizer 0.00001 --dataset halfcheetah
# python -m smooth.scripts.imitation_learning --epochs 70000 --n_train 10 --n_test 1000 --bs 1000 --lr 0.025 --heat_kernel_t 0.1 --clamp 0.09  --algorithm ERM --regularizer 0.00001 --dataset halfcheetah
