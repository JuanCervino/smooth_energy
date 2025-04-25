#!/bin/bash
#SBATCH --job-name imitation_smoothness
#SBATCH -o energy_imitation_%j.log
#SBATCH --gres=gpu:volta:1

# Initialize the module command first source
source /etc/profile

# Then use the module command to load the module needed for your work
module load anaconda/Python-ML-2024b
# module load anaconda/Python-ML-2023b # clu issue
# install this on the command line
# pip install --user -U torch-geometric

# pip install --user -U torch-sparse
# pip install --user -U torch-scatter

nvcc --version
nvidia-smi

# python -m smooth.scripts.imitation_learning --epochs 70000 --n_train 100 --n_test 1000 --bs 1000 --lr 0.1 --algorithm ERM
# python -m smooth.scripts.imitation_learning --epochs 70000 --n_train 100 --n_test 1000 --bs 1000 --lr 0.1 --heat_kernel_t 0.09 --clamp 0.9  --algorithm LaplacianRegularizationEuclidean --regularizer 0.001
# python -m smooth.scripts.imitation_learning --epochs 70000 --n_train 100 --n_test 1000 --bs 1000 --lr 0.1 --heat_kernel_t 0.1 --clamp 0.09  --algorithm LaplacianRegularizationMomentum --regularizer 0.00001
python --version
# python -m smooth.scripts.imitation_learning --epochs 1000 --n_train 100000 --n_test 1000 --bs 1024 --lr 0.025 --heat_kernel_t 0.1 --clamp 0.09  --algorithm LaplacianRegularizationEuclideanSparse --regularizer 0.00001 
python -m smooth.scripts.imitation_learning --epochs 70000 --n_train 10000 --n_test 1000 --bs 1024 --lr 0.025 --heat_kernel_t 0.1 --clamp 0.09  --algorithm LaplacianRegularizationEuclidean --regularizer 0.00001 
# python -m smooth.scripts.imitation_learning --epochs 40000 --n_train 40000 --n_test 1000 --bs 1024 --lr 0.025 --heat_kernel_t 0.1 --clamp 0.09  --algorithm LaplacianRegularizationEuclideanSparse --regularizer 0.00001 

# python -m smooth.scripts.imitation_learning --epochs 70000 --n_train 100 --n_test 1000 --bs 1000 --lr 0.04 --heat_kernel_t 0.1 --clamp 0.09  --algorithm MomentumGradient --regularizer 0.00001 
# python -m smooth.scripts.imitation_learning --epochs 70000 --n_train 1000 --n_test 1000 --bs 1000 --lr 0.02 --heat_kernel_t 0.09 --clamp 0.9  --algorithm ManifoldGradient --regularizer 0.00001
