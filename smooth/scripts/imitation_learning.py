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

# def get_lipschitz_constant(net, X):
#     """
#     Computes the Lipschitz constant of the network using the spectral norm.
#     Args:
#         net (nn.Module): The neural network model.
#         X (torch.Tensor): Input data to compute the Lipschitz constant.
#     Returns:
#         float: The Lipschitz constant of the network.
#     """
#     net.eval()  # Set the model to evaluation mode
#     with torch.no_grad():
#         X = X.to(next(net.parameters()).device)  # Ensure X is on the same device as the model
#         output = net(X)
#         # Compute the Jacobian matrix
#         jacobian = torch.autograd.functional.jacobian(net, X)
#         # Compute the spectral norm (largest singular value)
#         lipschitz_constant = torch.linalg.svdvals(jacobian).max().item()
#     return lipschitz_constant

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
            val_batch = torch.where(val_batch == 0, torch.tensor(1e-1, dtype=val_batch.dtype), val_batch)  # Avoid division by zero

            numerator = torch.abs (fx_row  - fx_col) # (batch_size, feat_dim)
            numerator = numerator.sum(dim=1)
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
    
    if X_unlab is not None:
        X_total = torch.concatenate((X_train, X_unlab), axis=0)
    else:
        X_total = X_train
    print('Ready to compute Laplacian of size', X_total.shape)
    print('Ready to compute Output of size', Y_train.shape)

    adj_matrix = laplacian.get_pairwise_distance_matrix(X_total, t=args.heat_kernel_t, distance_type='euclidean').to(device)
        # L = laplacian.get_laplacian(X_total, args.normalize, heat_kernel_t=args.heat_kernel_t, clamp_value = args.clamp).to(device)
    matrix = laplacian.get_knn_matrix(X_total,  distance_type = 'euclidean', matrix_type = 'knn', k=args.k, batch_size=200).to(device)

    
    dataset = NodeNeighborhoodDataset(matrix, X_total)
    unlabeled_loader_finite = DataLoader(dataset, batch_size=args.bs//args.k, shuffle=True, collate_fn=collate_fn)
    print('Completed Laplacian')


    print("Dataset loaded.")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")
    # Create NN
    net = FCNN3(input_dim = X_train.shape[-1],
                hidden_dim = args.hidden_neurons,
                num_classes = Y_train.shape[-1]).to(device)
    print(f"Model architecture: {net}")
    print(f"Number of parameters in the model: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")
        # Create the optimer
    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    
    
    
    if args.algorithm == 'ERM':

        columns = ['Epoch', 'Loss', 'Accuracy']
        utils.create_csv(args.output_dir, 'losses.csv', columns)

        for epoch in range(args.epochs):
            for i, data in enumerate(train_loader):
                inputs, labels = data
                optimizer.zero_grad()
                loss = F.mse_loss(net(inputs), labels)
                loss.backward()
                optimizer.step()


            if (epoch+1) % args.print_steps == 0 :
                with torch.no_grad():           
                    loss_test = mse_metric(net,test_loader,device)
                    loss_train = mse_metric(net,train_loader,device)
                utils.save_state(args.output_dir,epoch,loss_train,loss_test)
                print(epoch,loss_train,loss_test)
    
    elif args.algorithm == 'LaplacianRegularizationEuclidean':
        columns = ['Epoch', 'Loss','Regularized Laplacian Loss', 'Laplacian Loss', 'Accuracy']
        utils.create_csv(args.output_dir, 'losses.csv', columns)
        X_total = torch.concatenate((X_train, X_test), axis=0)
        L = laplacian.get_laplacian(X_total, args.normalize, heat_kernel_t=args.heat_kernel_t, clamp_value = args.clamp,distance_type = 'euclidean').to(device)

        e, V = np.linalg.eig(L.cpu().detach().numpy())
        print('Connected Components', np.sum(e < 0.0001))

        for epoch in range(args.epochs):
                    for i, data in enumerate(train_loader):
                        inputs, labels = data
                        optimizer.zero_grad()
                        f = net(inputs)
                        f_unlabel = net(X_total)

                        loss = F.mse_loss(f, labels)
                        loss_MSE = loss.item()
                        loss += args.regularizer * torch.trace(torch.matmul(f_unlabel.transpose(0,1),torch.matmul(L, f_unlabel)))
                        loss.backward()
                        optimizer.step()
                        # print(loss.item(),loss_MSE.item(),(loss-loss_MSE).item())
                    
                    if (epoch+1) % args.print_steps ==0:
                        with torch.no_grad():
                            loss_test = mse_metric(net,test_loader,device)
                            loss_train = mse_metric(net,train_loader,device)
                        utils.save_state(args.output_dir,epoch,loss_train,loss_train-loss_MSE,loss_MSE,loss_test)
                        print(f'Epoch {epoch}, loss {loss.item()}, loss diff {loss.item()-loss_MSE}, loss train {loss_train}, loss test {loss_test}')                        
                
    elif args.algorithm == 'LaplacianRegularizationEuclideanSparse':
        columns = ['Epoch', 'Loss','Regularized Laplacian Loss', 'Laplacian Loss', 'Accuracy']
        utils.create_csv(args.output_dir, 'losses.csv', columns)
        X_total = torch.concatenate((X_train, X_test), axis=0)
        print('Ready to compute Laplacian')
        start_time = time.time()
        matrix = laplacian.get_knn_matrix(X_total,  distance_type = 'euclidean', matrix_type = 'knn', k=args.k, batch_size=200)
        end_time = time.time()
        time_normal = end_time - start_time
        print('Time Normal', time_normal)

        dataset = NodeNeighborhoodDataset(matrix, X_total)
        
        # The batch size is k * batch_size
        loader = DataLoader(dataset, batch_size=args.bs//args.k, shuffle=True, collate_fn=collate_fn)
        unlabeled_loader = infinite_dataloader(loader)

        for epoch in range(args.epochs):
                    for i, data in enumerate(train_loader):
                        inputs, labels = data
                        optimizer.zero_grad()
                        f = net(inputs)

                        loss = F.mse_loss(f, labels)
                        loss_MSE = loss.item()
                        # row_batch, col_batch, val_batch = next(unlabeled_loader)
                        row_batch, col_batch, val_batch, x_row, x_col = next(unlabeled_loader)
                        
                        loss += args.regularizer * laplacian_quad_batch_from_features(net, x_row, x_col, val_batch)

                        loss.backward()
                        optimizer.step()
                        # print(loss.item(),loss_MSE.item(),(loss-loss_MSE).item())
                    
                    if (epoch+1) % args.print_steps ==0:
                        with torch.no_grad():
                            loss_test = mse_metric(net,test_loader,device)
                            loss_train = mse_metric(net,train_loader,device)
                        utils.save_state(args.output_dir,epoch,loss_train,loss_train-loss_MSE,loss_MSE,loss_test)
                        print(f'Epoch {epoch}, loss {loss.item()}, loss diff {loss.item()-loss_MSE}, loss train {loss_train}, loss test {loss_test}')                        
                
    elif args.algorithm == 'LaplacianRegularizationMomentum':
        columns = ['Epoch', 'Loss','Regularized Laplacian Loss', 'Laplacian Loss', 'Accuracy']
        utils.create_csv(args.output_dir, 'losses.csv', columns)
        print(args.heat_kernel_t, args.clamp, args.normalize)
        X_total = torch.concatenate((X_train, X_test), axis=0)

        L = laplacian.get_laplacian(X_total, args.normalize, heat_kernel_t=args.heat_kernel_t, clamp_value = args.clamp,distance_type = 'momentum_pendulum').to(device)

        e, V = np.linalg.eig(L.cpu().detach().numpy())
        print('Connected Components', np.sum(e < 0.0001))

        for epoch in range(args.epochs):
                    for i, data in enumerate(train_loader):
                        inputs, labels = data
                        optimizer.zero_grad()
                        f = net(inputs)
                        f_unlabel = net(X_total)
                        loss = F.mse_loss(f, labels)
                        loss_MSE = loss.item()
                        loss += args.regularizer * torch.trace(torch.matmul(f_unlabel.transpose(0,1),torch.matmul(L, f_unlabel)))
                        loss.backward()
                        optimizer.step()
                        # print(loss.item(),loss_MSE.item(),(loss-loss_MSE).item())
                    
                    if (epoch+1) % args.print_steps == 0 :
                        with torch.no_grad():
                            loss_test = mse_metric(net,test_loader,device)
                            loss_train = mse_metric(net,train_loader,device)
                        utils.save_state(args.output_dir,epoch,loss_train,loss_train-loss_MSE,loss_MSE,loss_test)
                        print(f'Epoch {epoch}, loss {loss.item()}, loss diff {loss.item()-loss_MSE}, loss train {loss_train}, loss test {loss_test}')                        
            



    elif args.algorithm == 'ManifoldGradientBatch':

            columns = ['Epoch', 'Loss CE','Regularized Laplacian Loss', 'Laplacian Loss', 'Accuracy']
            utils.create_csv(args.output_dir, 'losses.csv', columns)
            
            lambda_dual = torch.ones(len(X_total[:,0])) / len(X_total[:,0])  # Initialize dual variables for each sample in the dataset
            lambda_dual = lambda_dual.to(device).detach().requires_grad_(False)
            mu_dual = 5*torch.ones(1).to(device).detach().requires_grad_(False)
            # The batch size is k * batch_size
            unlabeled_loader_infinite = infinite_dataloader(unlabeled_loader_finite)
            # dataset_size = len(X_total[:,0])
            dataset_size = len(unlabeled_loader_finite.dataset)
            
            for epoch in range(args.epochs):
                # print(epoch)
                ############################################
                # Primal Update
                ############################################
                for i, data in enumerate(train_loader):
                        inputs, labels = data
                        optimizer.zero_grad()
                        f = net(inputs)

                        loss = F.mse_loss(f, labels)
                        loss_MSE = loss.item()

                        # Following samples uniformly from the unlabeled dataset
                            # row_batch, col_batch, val_batch, x_row, x_col = next(unlabeled_loader_infinite)
                        # Sample indices based on the dual variable
                        
                        probs = lambda_dual.cpu().numpy()
                        if not np.isclose(probs.sum(), 0):
                            non_zero_prob_indices = np.where(probs > 1e-9)[0]  # Using a small tolerance

                            # Determine the number of non-zero probabilities
                            num_non_zero_probs = len(non_zero_prob_indices)

                            # Calculate the sampling size as the minimum of non-zero probabilities and args.bs
                            sampling_size = min(num_non_zero_probs, args.bs)
                            # print("Sum is approximately zero.")
                        # else:
                            # print(f"Sum is {probs.sum()}, not zero.")
                            probs = probs / probs.sum()
                            samples = np.random.choice(np.arange(dataset_size), size=sampling_size, replace=False, p=probs)
                            # print('Samples', samples)
                            samples = torch.tensor(samples, dtype=torch.long, device=device)
                            batch = [unlabeled_loader_finite.dataset[idx] for idx in samples]
                            row_batch, col_batch, val_batch, x_row, x_col = collate_fn(batch)
                            
                            val_batch = val_batch * lambda_dual[row_batch].to(device)  # Apply dual variable to the values
                            loss += args.regularizer * laplacian_quad_batch_from_features(net, x_row, x_col, val_batch)

                        loss.backward()
                        optimizer.step()
                            # print(loss.item(),loss_MSE.item(),(loss-loss_MSE).item())
                ############################################
                # Dual Update
                ############################################
                if (epoch+1) % args.dual_update_steps ==0 :

                    with torch.no_grad():
                        for row_batch, col_batch, val_batch, x_row, x_col in unlabeled_loader_finite:

                            device = next(net.parameters()).device  # Get the device of the model
                            row_batch = row_batch.to(device)
                            col_batch = col_batch.to(device)
                            val_batch = val_batch.to(device)
                            val_batch = torch.where(val_batch == 0, torch.tensor(1e-1, dtype=val_batch.dtype), val_batch)  # Avoid division by zero

                            x_row = x_row.to(device)
                            x_col = x_col.to(device)
                            
                            fx_row = net(x_row)  # shape: (batch_size, feat_dim)
                            fx_col = net(x_col)  # shape: (batch_size, feat_dim)
                            numerator = torch.abs (fx_row  - fx_col).to(device)#.squeeze(1)  # (batch_size, feat_dim)
                            numerator = numerator.sum(dim=1)
                            division = torch.div(numerator, val_batch).to(device)  # (batch_size, feat_dim)
                            # Find unique keys and mapping
                            unique_rows, inverse_indices = torch.unique(row_batch, return_inverse=True)
                            unique_rows = unique_rows.to(device)  # Ensure it's on the same device
                            inverse_indices = inverse_indices.to(device)  # Ensure it's on the same device

                            # Initialize a tensor to hold maximums
                            max_values = torch.full((unique_rows.size(0),), float('-inf')).to(device)

                            # Scatter maximum
                            max_values = max_values.scatter_reduce(0, inverse_indices, division, reduce="amax", include_self=True)

                            lambda_dual[unique_rows] = torch.clamp(
                                                                    F.relu(lambda_dual[unique_rows] + args.dual_step * (max_values - args.lipschitz_constant)),
                                                                    max=args.clamp_dual
                                                                    )

                    #
                    # TODO: Normalize lambda_dual
                if (epoch+1) % args.print_steps ==0 :
                    with torch.no_grad():
                        acc = mse_metric(net,test_loader,device)
                        loss_train = mse_metric(net,train_loader,device)
                    utils.save_state(args.output_dir,epoch,loss_train,loss_train-loss_MSE,loss_MSE,acc)
                    lipschitz_constant = get_lipschitz_constant(net, unlabeled_loader_finite)  # Compute Lipschitz constant for logging

                    print(f'Epoch loss{epoch},\
                            loss {loss.item()},\
                            max lambda={torch.max(lambda_dual)}, \
                            positive lambdas={ (lambda_dual > 0).sum()}, \
                            lipschitz constant={ lipschitz_constant }, \
                            loss train {loss_train}, \
                            loss grad {loss.item()-loss_MSE}, \
                            loss MSE {loss_MSE}, \
                            loss test {acc}' )
       
    # Save final model
    lipschitz_constant = get_lipschitz_constant(net, unlabeled_loader_finite)  # Compute Lipschitz constant for logging
    print(f'Lipschitz constant: {lipschitz_constant}')
    print('Saving final model...')
    model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save(net.state_dict(), model_path)
    print(f'Model saved to {model_path}')

              
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Manifold Regularization with Synthetic Data')

    parser.add_argument('--output_dir', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='inverted_pendulum')
    parser.add_argument('--n_dim', type=int, default=2, help='Dimension')
    parser.add_argument('--n_train', type=int, default=1)
    parser.add_argument('--n_unlab', type=int, default=0, help='Number of samples per class')
    parser.add_argument('--n_test', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default='./smooth/data')
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--trajectory_length', type=int, default=300, help='Length of the trajectory for the dataset')

    parser.add_argument('--print_steps', type=int, default=250)
    parser.add_argument('--sampling', type=str, default='trajectory', help='Sampling method for the dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')


    parser.add_argument('--algorithm', type=str, default='ERM')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dual_update_steps', type=int, default=100)

    parser.add_argument('--regularizer', type=float, default=1e-6)
    parser.add_argument('--heat_kernel_t', type=float, default=1e-2)
    parser.add_argument('--normalize', type=bool, default=True)

    parser.add_argument('--hidden_neurons', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)

    parser.add_argument('--dual_step', type=float, default=0.5)
    parser.add_argument('--clamp', type=float, default=0.1)
    parser.add_argument('--clamp_dual', type=float, default=1000.0, help='Clamp the dual variables to this value')
    parser.add_argument('--lipschitz_constant', type=float, default=1)

    parser.add_argument('--k', type=int, default=10)

    args = parser.parse_args()

    args.output_dir = args.output_dir + '/' + str(args.dataset) +  '/' + args.algorithm+  '/'  + datetime.now().strftime("%Y-%m%d-%H%M%S-%f")
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    
    # Create logs directory and save logs
    logs_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, 'log.txt')
    sys.stdout = Logger(log_file)

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # if args.dataset not in vars(datasets):
    #     raise NotImplementedError(f'Dataset {args.dataset} is not implemented.')

    main(args)