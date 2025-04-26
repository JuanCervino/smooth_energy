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

    print('Loading dataset...')
    # Load dataset
    if args.sampling == 'trajectory':
        data = imitation.create_imitation_dataset_trajectories(args.dataset,
                                               n_train=args.n_train,
                                               n_unlab=args.n_unlab,
                                               n_test=args.n_test,
                                               T=args.trajectory_length)
    else:
        data = imitation.create_imitation_dataset(args.dataset,
                                               ids_train=np.arange(args.n_train),
                                               ids_unlab=np.arange(args.n_train, args.n_train + args.n_unlab),
                                               ids_test=np.arange(args.n_train + args.n_unlab, args.n_train + args.n_unlab + args.n_test))
    X_train, Y_train, X_unlab, y_unlab, X_test, Y_test = data

    # Create Dataset
    train_dataset = RegressionDataset(X_train, Y_train)
    test_dataset = RegressionDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs)
    print("Dataset loaded.")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")
    # Create N
    # Suppose later you want to load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(args.output_dir, args.dataset, args.algorithm, args.experiment_date, f'final_model.pth')
    # model_path = 'final_model.pth'
    model = load_model(FCNN3, model_path, device)

    # Now you can do inference
    with torch.no_grad():
        acc = mse_metric(model,test_loader,device)
        print(f'Test MSE: {acc:.4f}')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Manifold Regularization with Synthetic Data')

    parser.add_argument('--dataset', type=str, default='inverted_pendulum')
    parser.add_argument('--n_dim', type=int, default=2, help='Dimension')
    parser.add_argument('--n_train', type=int, default=1)
    parser.add_argument('--n_unlab', type=int, default=0, help='Number of samples per class')
    parser.add_argument('--n_test', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='test')
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--trajectory_length', type=int, default=300, help='Length of the trajectory for the dataset')
    parser.add_argument('--sampling', type=str, default='trajectory', help='Sampling method for the dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--algorithm', type=str, default='ERM')
    parser.add_argument('--experiment_date', type=str, default='2025-0425-175227', help='Date of the experiment')

    args = parser.parse_args()
    main(args)