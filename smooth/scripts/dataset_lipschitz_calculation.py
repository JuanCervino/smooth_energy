import torch
import torch.nn as nn
import os   
import argparse
from datetime import datetime
import numpy as np
from smooth.lib import utils
from smooth.lib import imitation
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import json
from smooth import laplacian
from torch_sparse import SparseTensor
import time # Delete this
import random
import sys
import matplotlib.pyplot as plt

# TODO: move all this code out of here
# TODO: add the threshold for the laplacian


def load_model(model_class, model_path, device='cpu'):
    """
    model_class: the class of the model (e.g., FCNN3)
    model_path: path to the .pth file
    device: 'cpu' or 'cuda'
    """
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Important: set to eval mode for inference
    return model

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
class RegressionDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X[:,0])

    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx,:]

class SparseCOODataset(Dataset):
    def __init__(self, sparse_tensor: SparseTensor):
        self.row, self.col, self.val = sparse_tensor.coo()
        self.length = self.row.size(0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.row[idx], self.col[idx], self.val[idx]


def infinite_dataloader(dataloader):
    """Creates an infinite generator from a DataLoader"""
    while True:
        for batch in dataloader:
            yield batch
            
# def collate_fn(batch):
#     rows, cols, vals = zip(*batch)
#     return torch.stack(rows), torch.stack(cols), torch.stack(vals)

# def collate_fn(batch):
#     r, c, v, x_r, x_c = zip(*batch)
#     return (torch.stack(r),
#             torch.stack(c),
#             torch.stack(v),
#             torch.stack(x_r),
#             torch.stack(x_c))

def get_lipschitz_constant(net, unlabeled_loader_finite):
    """
    Computes the Lipschitz constant of the network using the spectral norm.
    Args:
        net (nn.Module): The neural network model.
        X (torch.Tensor): Input data to compute the Lipschitz constant.
    Returns:
        float: The Lipschitz constant of the network.
    """
    lipschitz_constant = 0.0
    net.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for row_batch, col_batch, val_batch, x_row, x_col in unlabeled_loader_finite:
            fx_row = net(x_row)  # shape: (batch_size, feat_dim)
            fx_col = net(x_col)  # shape: (batch_size, feat_dim)
            device = fx_row.device  # Get the device of the model
            
            # Ensure all tensors are on the same device
            row_batch = row_batch.to(device)
            col_batch = col_batch.to(device)
            val_batch = val_batch.to(device)
            
            numerator = torch.abs (fx_row  - fx_col).squeeze(1)  # (batch_size, feat_dim)
            division = torch.div(numerator, val_batch)
            # Find unique keys and mapping
            unique_rows, inverse_indices = torch.unique(row_batch, return_inverse=True)
            unique_rows = unique_rows.to(device)  # Ensure it's on the same device
            inverse_indices = inverse_indices.to(device)  # Ensure it's on the same device

            # Initialize a tensor to hold maximums
            max_values = torch.full((unique_rows.size(0),), float('-inf'), device=device)
            max_values = max_values.scatter_reduce(0, inverse_indices, division, reduce="amax", include_self=True)
            if max_values.max().item() > lipschitz_constant:
                lipschitz_constant = max_values.max().item()
    return lipschitz_constant


def get_lipschitz_constant_from_data(Y_train, unlabeled_loader_finite):
    """
    Computes the Lipschitz constant of the network using the spectral norm.
    Args:
        net (nn.Module): The neural network model.
        X (torch.Tensor): Input data to compute the Lipschitz constant.
    Returns:
        float: The Lipschitz constant of the network.
    """
    lipschitz_constant = 0.0
    with torch.no_grad():
        for row_batch, col_batch, val_batch, x_row, x_col in unlabeled_loader_finite:
            # fx_row = net(x_row)  # shape: (batch_size, feat_dim)
            # fx_col = net(x_col)  # shape: (batch_size, feat_dim)
            fx_row = Y_train[row_batch]  # shape: (batch_size, feat_dim)
            fx_col = Y_train[col_batch]  # shape: (batch_size, feat_dim)
            
            device = fx_row.device  # Get the device of the model
            
            # Ensure all tensors are on the same device
            row_batch = row_batch.to(device)
            col_batch = col_batch.to(device)
            val_batch = val_batch.to(device)
            val_batch = torch.where(val_batch == 0, torch.tensor(1e-1, dtype=val_batch.dtype), val_batch)  # Avoid division by zero

            numerator = torch.abs (fx_row  - fx_col).squeeze(1)  # (batch_size, feat_dim)
            division = torch.div(numerator, val_batch)
            # Find unique keys and mapping
            unique_rows, inverse_indices = torch.unique(row_batch, return_inverse=True)
            unique_rows = unique_rows.to(device)  # Ensure it's on the same device
            inverse_indices = inverse_indices.to(device)  # Ensure it's on the same device

            # Initialize a tensor to hold maximums
            max_values = torch.full((unique_rows.size(0),), float('-inf'), device=device)
            max_values = max_values.scatter_reduce(0, inverse_indices, division, reduce="amax", include_self=True)
            if max_values.max().item() > lipschitz_constant:
                lipschitz_constant = max_values.max().item()
    return lipschitz_constant


class NodeNeighborhoodDataset(Dataset):
    def __init__(self, sparse_tensor, X):
        self.row, self.col, self.val = sparse_tensor.coo()
        self.X = X  # (num_nodes, feat_dim)

        # Preprocess: for each node, store indices where it's a row
        self.num_nodes = X.size(0)
        self.row_to_edge_indices = [[] for _ in range(self.num_nodes)]
        for idx in range(self.row.size(0)):
            n = self.row[idx].item()
            self.row_to_edge_indices[n].append(idx)

        # Filter to nodes that actually have outgoing edges
        self.valid_nodes = [i for i, e in enumerate(self.row_to_edge_indices) if len(e) > 0]

    def __len__(self):
        return len(self.valid_nodes)

    def __getitem__(self, idx):
        node = self.valid_nodes[idx]
        edge_indices = self.row_to_edge_indices[node]

        rows = self.row[edge_indices]           # Should all be node
        cols = self.col[edge_indices]
        vals = self.val[edge_indices]
        x_row = self.X[rows]                    # Repeated node features
        x_col = self.X[cols]                    # Neighbor features

        return rows, cols, vals, x_row, x_col

def laplacian_quad_batch_from_features(f, x_row, x_col, val_batch):
    fx_row = f(x_row)  # shape: (batch_size, feat_dim)
    fx_col = f(x_col)  # shape: (batch_size, feat_dim)
    diff = fx_col - fx_row
    quad = (diff**2).sum(dim=-1) * val_batch  # (batch_size,)
    return quad.sum()

def collate_fn(batch):
    # Each item in batch is (rows, cols, vals, x_row, x_col) of varying length
    rows, cols, vals, x_rows, x_cols = zip(*batch)

    return (
        torch.cat(rows),
        torch.cat(cols),
        torch.cat(vals),
        torch.cat(x_rows),
        torch.cat(x_cols)
    )

@torch.no_grad()
def accuracy(net, loader, device):
    
    correct, total = 0, 0
    net = net.to(device)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        output = net(imgs).to(device)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.size(0)
    return 100. * correct / total

@torch.no_grad()
def mse_metric(net, loader, device):
    total_mse = 0.0
    total_samples = 0
    net = net.to(device)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        output = net(imgs).to(device)
        mse = F.mse_loss(output, labels, reduction='sum')  # Compute MSE for the batch
        total_mse += mse.item()
        total_samples += labels.size(0)
    return total_mse / total_samples

class FCNN3(nn.Module):
    def __init__(self, input_dim = 4, hidden_dim = 32, num_classes = 1):
        super(FCNN3, self).__init__()

        self.layer1 = nn.Linear(in_features=input_dim,out_features=hidden_dim,bias=True)
        self.layer2 = nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=True)
        self.layer3 = nn.Linear(in_features=hidden_dim,out_features=num_classes,bias=False)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        out = self.layer3(x)
        return out


def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    set_seed(args.seed)

    n_trains = [ 2, 4, 6, 8, 10, 12, 14, 16, 18, 20] #, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
    n_unlab = 0
    n_test = 2
    lipschitz_constants = []
    for n_train in n_trains:
        print('Loading dataset...')
        # Load dataset
        if args.sampling == 'trajectory':
            data = imitation.create_imitation_dataset_trajectories(args.dataset,
                                                n_train=n_train,
                                                n_unlab=n_unlab,
                                                n_test=n_test,
                                                T=args.trajectory_length)
        else:
            data = imitation.create_imitation_dataset(args.dataset,
                                                ids_train=np.arange(n_train),
                                                ids_unlab=np.arange(n_train, n_train + n_unlab),
                                                ids_test=np.arange(n_train + n_unlab, n_train + n_unlab + n_test))
        X_train, Y_train, X_unlab, y_unlab, X_test, Y_test = data
    
        if X_unlab is not None:
            X_total = torch.concatenate((X_train, X_unlab), axis=0)
        else:
            X_total = X_train
        # Create Dataset
        # train_dataset = RegressionDataset(X_train, Y_train)
        # test_dataset = RegressionDataset(X_test, Y_test)
        # train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        # test_loader = DataLoader(test_dataset, batch_size=args.bs)
        
        print('Ready to compute Laplacian of size', X_total.shape)
        adj_matrix = laplacian.get_pairwise_distance_matrix(X_total, t=args.heat_kernel_t, distance_type='euclidean').to(device)
            # L = laplacian.get_laplacian(X_total, args.normalize, heat_kernel_t=args.heat_kernel_t, clamp_value = args.clamp).to(device)
        matrix = laplacian.get_knn_matrix(X_total,  distance_type = 'euclidean', matrix_type = 'knn', k=args.k, batch_size=200).to(device)
        dataset = NodeNeighborhoodDataset(matrix, X_total)
        unlabeled_loader_finite = DataLoader(dataset, batch_size=args.bs//args.k, shuffle=True, collate_fn=collate_fn)
        
        # Save final model
        lipschitz_constant = get_lipschitz_constant_from_data(Y_train, unlabeled_loader_finite)  # Compute Lipschitz constant for logging
        print(f'Lipschitz constant: {lipschitz_constant}')
        lipschitz_constants.append(lipschitz_constant)
    
    plt.plot(n_trains, lipschitz_constants, marker='o')
    plt.xlabel('Number of training samples')
    plt.ylabel('Lipschitz constant')
    plt.title('Lipschitz constant vs. number of training samples')
    plt.grid()
    plt.savefig(f'lipschitz_constant_{args.dataset}_{args.sampling}.png')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate Lipschitz constant for a model on a dataset')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--sampling', type=str, default='trajectory', help='Sampling method for the dataset')
    parser.add_argument('--dataset', type=str, default='inverted_pendulum')
    parser.add_argument('--trajectory_length', type=int, default=300, help='Length of the trajectory for the dataset')
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--heat_kernel_t', type=float, default=1e-2)
    parser.add_argument('--k', type=int, default=10)

    args = parser.parse_args()
    main(args)