
python -m smooth.scripts.imitation_learning --epochs 1000 --n_train 100 --n_test 100 --bs 32 --lr 0.025 --heat_kernel_t 0.1 --clamp 0.09  --algorithm ManifoldGradientBatch --regularizer 0.00001 
