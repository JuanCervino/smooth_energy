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

class RegressionDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X[:,0])

    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx,:]


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
    def __init__(self, input_dim = 4, hidden_dim = 64, num_classes = 1):
        super(FCNN3, self).__init__()

        self.layer1 = nn.Linear(in_features=input_dim,out_features=hidden_dim,bias=True)
        self.layer2 = nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=True)
        self.layer3 = nn.Linear(in_features=hidden_dim,out_features=num_classes,bias=False)

    def forward(self, x):
        # out = F.relu(self.layer1(x))
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        out = self.layer3(x)
        return out

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    print('Loading dataset...')
    # Load dataset
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
    # Create NN
    net = FCNN3().to(device)
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


            if epoch % 100==0:
                with torch.no_grad():           
                    loss_test = mse_metric(net,test_loader,device)
                    loss_train = mse_metric(net,train_loader,device)
                utils.save_state(args.output_dir,epoch,loss_train,loss_test)
                print(epoch,loss_train,loss_test)
    
    elif args.algorithm == 'LaplacianRegularizationEuclidean':
        columns = ['Epoch', 'Loss','Regularized Laplacian Loss', 'Laplacian Loss', 'Accuracy']
        utils.create_csv(args.output_dir, 'losses.csv', columns)
        L = laplacian.get_laplacian(X_train, args.normalize, heat_kernel_t=args.heat_kernel_t, clamp_value = args.clamp,distance_type = 'euclidean').to(device)

        e, V = np.linalg.eig(L.cpu().detach().numpy())
        print('Connected Components', np.sum(e < 0.0001))

        for epoch in range(args.epochs):
                    for i, data in enumerate(train_loader):
                        inputs, labels = data
                        optimizer.zero_grad()
                        f = net(inputs)
                        loss = F.mse_loss(f, labels)
                        loss_MSE = loss.item()
                        loss += args.regularizer * torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f)))
                        loss.backward()
                        optimizer.step()
                        # print(loss.item(),loss_MSE.item(),(loss-loss_MSE).item())
                    
                    if epoch % 100 ==0:
                        with torch.no_grad():
                            loss_test = mse_metric(net,test_loader,device)
                            loss_train = mse_metric(net,train_loader,device)
                        utils.save_state(args.output_dir,epoch,loss_train,loss_train-loss_MSE,loss_MSE,loss_test)
                        print(f'Epoch {epoch}, loss {loss.item()}, loss diff {loss.item()-loss_MSE}, loss train {loss_train}, loss test {loss_test}')                        
                
    elif args.algorithm == 'LaplacianRegularizationMomentum':
        columns = ['Epoch', 'Loss','Regularized Laplacian Loss', 'Laplacian Loss', 'Accuracy']
        utils.create_csv(args.output_dir, 'losses.csv', columns)
        L = laplacian.get_laplacian(X_train, args.normalize, heat_kernel_t=args.heat_kernel_t, clamp_value = args.clamp,distance_type = 'momentum_pendulum').to(device)

        e, V = np.linalg.eig(L.cpu().detach().numpy())
        print('Connected Components', np.sum(e < 0.0001))

        for epoch in range(args.epochs):
                    for i, data in enumerate(train_loader):
                        inputs, labels = data
                        optimizer.zero_grad()
                        f = net(inputs)
                        loss = F.mse_loss(f, labels)
                        loss_MSE = loss.item()
                        loss += args.regularizer * torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f)))
                        loss.backward()
                        optimizer.step()
                        # print(loss.item(),loss_MSE.item(),(loss-loss_MSE).item())
                    
                    if epoch % 100 ==0:
                        with torch.no_grad():
                            loss_test = mse_metric(net,test_loader,device)
                            loss_train = mse_metric(net,train_loader,device)
                        utils.save_state(args.output_dir,epoch,loss_train,loss_train-loss_MSE,loss_MSE,loss_test)
                        print(f'Epoch {epoch}, loss {loss.item()}, loss diff {loss.item()-loss_MSE}, loss train {loss_train}, loss test {loss_test}')                        
            
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Manifold Regularization with Synthetic Data')

    parser.add_argument('--output_dir', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='inverted_pendulum')
    parser.add_argument('--n_dim', type=int, default=2, help='Dimension')
    parser.add_argument('--n_train', type=int, default=1)
    parser.add_argument('--n_unlab', type=int, default=100, help='Number of samples per class')
    parser.add_argument('--n_test', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default='./smooth/data')
    parser.add_argument('--bs', type=int, default=8)


    parser.add_argument('--algorithm', type=str, default='ERM')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--regularizer', type=float, default=1e-6)
    parser.add_argument('--heat_kernel_t', type=float, default=1e-2)
    parser.add_argument('--normalize', type=bool, default=True)

    parser.add_argument('--hidden_neurons', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.9)

    parser.add_argument('--dual_step_mu', type=float, default=0.5)
    parser.add_argument('--dual_step_lambda', type=float, default=0.1)
    parser.add_argument('--rho_step', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--clamp', type=float, default=0.1)

    args = parser.parse_args()

    args.output_dir = args.output_dir + '/' + str(args.dataset) +  '_' + args.algorithm+  '_'  + datetime.now().strftime("%Y-%m%d-%H%M%S")
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # if args.dataset not in vars(datasets):
    #     raise NotImplementedError(f'Dataset {args.dataset} is not implemented.')

    main(args)