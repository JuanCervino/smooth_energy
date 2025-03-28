from smooth import laplacian_old as laplacian
# from smooth import laplacian

from smooth.lib import toyexample
import torch
import json
import scipy.io
import numpy as np
import argparse
import os
import sys
import time
from smooth.lib import utils

import matplotlib.pyplot as plt

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data0 = np.load(args.data_dir+'/rad0.50_data_filtered_0.npy')
    data1 = np.load(args.data_dir+'/rad0.50_data_filtered_1.npy')

    X_train = torch.tensor(data0[:-1,:6,0]).float().to(device) # Do no get accelerations
    Y_train = torch.tensor(data0[:-1,6:,0]).float().to(device)  # Remove gravity
    Y_train[:,2] = Y_train[:,2] - 9.81  # Remove accelerations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_train[:,0], X_train[:,1], X_train[:,2], c=np.ones_like(X_train[:,2]), cmap='viridis', s=100)

    # Add labels and color bar
    ax.set_xlabel('X (i)')
    ax.set_ylabel('Y (j)')
    ax.set_zlabel('Z (k)')
    fig.colorbar(scatter, ax=ax, label='Value')

    flat = Y_train.flatten(start_dim = 1)
    A = torch.cdist(flat,flat)
    plt.figure()
    plt.imshow(A, cmap='viridis')
    plt.colorbar()  # Add a colorbar to show the mapping of values to colors
    plt.title("Data Visualization")
    plt.savefig(f'data_output.png')
    

    plt.title('3D Sequence Positions and Values')
    plt.savefig('3D_sequence_positions.png')
    # Y_train = torch.tensor(data0[1:,0:6,0]).float().to(device)
    # X_test = torch.tensor(data1[:-1,:,0]).float().to(device)
    # Y_test = torch.tensor(data1[1:,0:6,0]).float().to(device)



    if args.plot_adjacency:
        plt.figure()
        A = laplacian.get_pairwise_distance_matrix(X_train, args.heat_kernel_t, distance_type = args.distance_type).to(device)
        # print('A adj_matrix', A.shape, np.diag(A))
        print('Adjacency Values', A[:10,:10])

        plt.imshow(A, cmap='viridis')
        plt.colorbar()  # Add a colorbar to show the mapping of values to colors
        plt.title("Adjancency Visualization")
        plt.savefig(f'adj_matrix_{args.distance_type}_{args.heat_kernel_t}.png')
    
    if args.plot_laplacian:
        L = laplacian.get_laplacian(X_train, args.normalize, heat_kernel_t=args.heat_kernel_t, clamp_value = args.clamp, distance_type = args.distance_type).to(device)

        plt.figure()
        print('Laplacian Values', L[:10,:10])
        plt.imshow(L, cmap='viridis')
        plt.colorbar()  # Add a colorbar to show the mapping of values to colors
        plt.title(f"Matrix Visualization_{args.distance_type}_{args.heat_kernel_t}_{args.clamp}")
        plt.savefig(f'laplacian_matrix_{args.distance_type}_{args.heat_kernel_t}_{args.clamp}.png')
        
      


        plt.figure()
        print('Laplacian Values', L[:10,:10])
        L = L.fill_diagonal_(0)
        plt.imshow(L, cmap='viridis')
        plt.colorbar()  # Add a colorbar to show the mapping of values to colors
        plt.title(f"Zero Diag Matrix Visualization_{args.distance_type}_{args.heat_kernel_t}_{args.clamp}")
        plt.savefig(f'Zero Diag Matrix Visualization{args.distance_type}_{args.heat_kernel_t}_{args.clamp}.png')
        
      


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Laplacian')
    parser.add_argument('--normalize', action='store_true', help='Enable normalization')
    parser.add_argument('--clamp', type=float, default=0.3)
    parser.add_argument('--heat_kernel_t', type=float, default=0.5)
    parser.add_argument('--data_dir', type=str, default='./smooth/data')
    parser.add_argument('--distance_type', type=str, default='euclidean')
    parser.add_argument('--plot_laplacian', type=bool, default=True)
    parser.add_argument('--plot_adjacency', type=bool, default=False)

    args = parser.parse_args()
    print('args', args)
    main(args)