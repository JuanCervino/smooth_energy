# For Two Moons Dataset
# 2 Connected Components
# python -m smooth.scripts.toy_example_manifold  --algorithm ERM\
#                        --n_train 1 --n_unlab 200 --epochs 15000\
#                         --hidden_neurons 64 --weight_decay 0. --lr 0.9\
#                         --regularizer 0. --heat_kernel_t 0.005\
#                         --normalize True --noise 0.05

# python -m smooth.scripts.toy_example_manifold  --algorithm ERM\
#                        --n_train 1 --n_unlab 200 --epochs 15000\
#                         --hidden_neurons 64 --weight_decay 0.1 --lr 0.9\
#                         --regularizer 0. --heat_kernel_t 0.005\
#                         --normalize True --noise 0.05

# python -m smooth.scripts.toy_example_manifold  --algorithm LAPLACIAN_REGULARIZATION\
#                        --n_train 1 --n_unlab 200 --epochs 15000\
#                         --hidden_neurons 64 --weight_decay 0. --lr 0.9\
#                         --regularizer 0.5 --heat_kernel_t 0.005\
#                         --normalize True --noise 0.05

# python -m smooth.scripts.toy_example_manifold  --algorithm MANIFOLD_GRADIENT_NO_RHO\
#                        --n_train 1 --n_unlab 200 --epochs 15000\
#                         --hidden_neurons 64 --weight_decay 0. --lr 2\
#                         --regularizer 1 --heat_kernel_t 0.005\
#                         --normalize True --noise 0.05 --epsilon 0.02


# # 1 Connected Components
# python -m smooth.scripts.toy_example_manifold  --algorithm MANIFOLD_GRADIENT_NO_RHO\
#                         --n_train 1 --n_unlab 200 --epochs 15000\
#                         --hidden_neurons 64 --weight_decay 0. --lr .9\
#                         --regularizer 1 --heat_kernel_t 0.005\
#                         --normalize True --noise 0.1 --epsilon 0.2


# python -m smooth.scripts.toy_example_manifold  --algorithm MANIFOLD_GRADIENT_NO_RHO\
#                         --n_train 1 --n_unlab 200 --epochs 15000\
#                         --hidden_neurons 64 --weight_decay 0. --lr .9\
#                         --regularizer 1 --heat_kernel_t 0.019\
#                         --normalize True --noise 0.1 --epsilon 0.2





python -m smooth.scripts.quadrotor_state_prediction --algorithm ERM\
                        --epochs 10000 --bs 5999\
                        --hidden_neurons 8192 --weight_decay 0.2 --lr 0.00001\
                        --regularizer 0.000001 --heat_kernel_t 0.004\
                        --normalize True --epsilon 0.1 --weight_decay 0. --heat_kernel_t 0.01


# python -m smooth.scripts.quadrotor_state_prediction --algorithm ERM\
#                         --epochs 10000 --bs 5999\
#                         --hidden_neurons 8192 --weight_decay 0.2 --lr 0.00001\
#                         --regularizer 0.000001 --heat_kernel_t 0.004\
#                         --normalize True --epsilon 0.1 --weight_decay 0.1 --heat_kernel_t 0.01

# python -m smooth.scripts.quadrotor_state_prediction --algorithm LAPLACIAN_REGULARIZATION\
#                         --epochs 10000 --bs 5999\
#                         --hidden_neurons 8192 --weight_decay 0. --lr 0.00001\
#                         --regularizer 0.000001\
#                         --normalize True --epsilon 0.1\
#                         --weight_decay 0. --heat_kernel_t 0.01

# python -m smooth.scripts.quadrotor_state_prediction --algorithm MANIFOLD_GRADIENT_NO_RHO\
#                         --epochs 10000 --bs 5999\
#                         --hidden_neurons 8192 --weight_decay 0. --lr 0.00001\
#                         --regularizer 0.000001\
#                         --normalize True --epsilon 0.03\
#                         --weight_decay 0. --heat_kernel_t 0.01

