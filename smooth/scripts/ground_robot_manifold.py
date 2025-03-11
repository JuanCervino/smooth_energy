import torchvision.models as models
import torchvision.transforms as transforms

import torch.optim as optim
import time
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import argparse, os
import torch.nn.functional as F
from torch import nn

from smooth import laplacian
from smooth.lib import toyexample
import torch
import json
import scipy.io
import numpy as np
from smooth.lib import utils

class RegressionDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X[:,0])

    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx,:], idx


@torch.no_grad()
def accuracy(net, loader, device):
    correct, total = 0, 0
    net = net.to(device)
    for imgs, labels,_ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        output = net(imgs).to(device)
        correct += F.mse_loss(output,labels).sum().item()
        total += imgs.size(0)
    return correct 

class FCNN(nn.Module):
    def __init__(self, input_dim = 98, hidden_dim = 256, num_classes = 2):
        super(FCNN, self).__init__()

        self.layer1 = nn.Linear(in_features=input_dim,out_features=hidden_dim,bias=True)
        self.layer2 = nn.Linear(in_features=hidden_dim,out_features=num_classes,bias=False)

    def forward(self, x):
        # out = F.relu(self.layer1(x))
        out = torch.tanh(self.layer1(x))
        out = self.layer2(out)
        return out
    
    
def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Dataset
    if args.dataset == 'ground_robot_pavement':
        matP = scipy.io.loadmat(args.data_dir+"/olda_pavement.mat")
        # X = torch.zeros([49,2,224])
        # X[:,0,:] = torch.tensor(matP['X'][0:49,:])
        # X[:,1,:] = torch.tensor(matP['X'][49:49+49,:])
        X_train = torch.tensor(matP['X'][0:98,0:200]).transpose(0,1).float().to(device)
        X_test = torch.tensor(matP['X'][0:98,200:224]).transpose(0,1).float().to(device)

        Y_train = torch.tensor((matP['Y'][:,0:200])).transpose(0,1).float().to(device)
        Y_test = torch.tensor((matP['Y'][:,200:224])).transpose(0,1).float().to(device)
        
    elif args.dataset == 'ground_robot_grass':
        matG = scipy.io.loadmat(args.data_dir+"/olda_grass.mat")
        # X = torch.zeros([49,2,195])
        # X[:,0,:] = torch.tensor(matG['X'][0:49,:])
        # X[:,1,:] = torch.tensor(matG['X'][49:49+49,:])

        X_train = torch.tensor(matG['X'][0:98,0:170]).transpose(0,1).float().to(device)
        X_test = torch.tensor(matG['X'][0:98,170:195]).transpose(0,1).float().to(device)
        Y_train = torch.tensor((matG['Y'][0:98,0:170])).transpose(0,1).float().to(device)
        Y_test = torch.tensor((matG['Y'][0:98,170:195])).transpose(0,1).float().to(device)

    # Create Dataset
    train_dataset = RegressionDataset(X_train, Y_train)
    test_dataset = RegressionDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs)
    # Create NN
    net = FCNN().to(device)

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
                inputs, labels, idx = data
                optimizer.zero_grad()
                loss = F.mse_loss(net(inputs), labels)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                acc = accuracy(net,test_loader,'cuda')
            loss_train = accuracy(net,train_loader,'cuda')
            utils.save_state(args.output_dir,epoch,loss_train,acc)
            if epoch % 100==0:
                print(epoch,loss_train,acc)

    if args.algorithm == 'LAPLACIAN_REGULARIZATION':
        columns = ['Epoch', 'Loss','Regularized Laplacian Loss', 'Laplacian Loss', 'Accuracy']
        utils.create_csv(args.output_dir, 'losses.csv', columns)
        L = laplacian.get_laplacian(X_train, args.normalize, heat_kernel_t=args.heat_kernel_t, clamp_value = args.clamp).to(device)

        e, V = np.linalg.eig(L.cpu().detach().numpy())
        print('Connected Components', np.sum(e < 0.0001))
        
        for epoch in range(args.epochs):
            for i, data in enumerate(train_loader):
                inputs, labels, idx = data
                optimizer.zero_grad()
                f = net(inputs)
                loss = F.mse_loss(f, labels)
                loss_MSE = loss.item()
                loss += args.regularizer * torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f)))
                loss.backward()
                optimizer.step()
                # print(loss.item(),loss_MSE.item(),(loss-loss_MSE).item())
            with torch.no_grad():
                acc = accuracy(net,test_loader,'cuda')
            loss_train = accuracy(net,train_loader,'cuda')
            utils.save_state(args.output_dir,epoch,loss_train,loss_train-loss_MSE,loss_MSE,acc)
            if epoch % 100 ==0:
                print(epoch,loss.item(),loss.item()-loss_MSE,loss_train,acc)                        
    if args.algorithm == 'MANIFOLD_GRADIENT_NO_RHO':

        columns = ['Epoch', 'Loss CE','Regularized Laplacian Loss', 'Laplacian Loss', 'Accuracy']
        utils.create_csv(args.output_dir, 'losses.csv', columns)
        L = laplacian.get_laplacian(X_train, args.normalize, heat_kernel_t=args.heat_kernel_t, clamp_value = args.clamp).to(device)
        adj_matrix = torch.cdist(X_train, X_train)
        print(L)
        e, V = np.linalg.eig(L.cpu().detach().numpy())
        print('Connected Components', np.sum(e < 0.0001))
        lambda_dual = torch.ones(len(Y_train)) / len(Y_train)
        lambda_dual = lambda_dual.to(device).detach().requires_grad_(False)
        mu_dual = 5*torch.ones(1).to(device).detach().requires_grad_(False)
        for epoch in range(args.epochs):
            # print(epoch)
            ############################################
            # Primal Update
            ############################################
            for i, data in enumerate(train_loader):
                inputs, labels, idx = data
                optimizer.zero_grad()
                f = net(inputs)
                loss = mu_dual * F.mse_loss(f, labels)
                loss_MSE = loss.item()
                loss += args.regularizer * torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f)))
                loss.backward()
                optimizer.step()
                        # print(loss.item(),loss_MSE.item(),(loss-loss_MSE).item())
            ############################################
            # Dual Update
            ############################################
            with torch.no_grad():
                # mu_dual = torch.nn.functional.relu(mu_dual + args.dual_step_mu * (F.cross_entropy(net(X_lab), y_lab) - args.epsilon))
                mu_dual = torch.clamp(mu_dual + args.dual_step_mu * (F.mse_loss(net(X_train), Y_train) - args.epsilon),0,5)
                f_prime = net(X_train)
                f_matrix= []
                f_matrix.append([])
                f_matrix [0] = torch.cat([f_prime[:,0]] * f_prime.shape[0]).reshape((f_prime.shape[0], f_prime.shape[0]))
                f_matrix.append([])
                f_matrix[1] = torch.cat([f_prime[:,1]] * f_prime.shape[0]).reshape((f_prime.shape[0], f_prime.shape[0]))

                numerator = torch.abs (f_matrix [0] - f_matrix[0].transpose(0,1)) + torch.abs(f_matrix [1] - f_matrix[1].transpose(0,1)).to(device)
                division = torch.div(numerator, (adj_matrix + torch.eye(f_prime.shape[0]).to(device)))
                [grads,indices] = torch.max(division, 1)
                # grads = grads.pow(2)
                # print(grads)
                lambda_dual = F.relu(lambda_dual + args.dual_step_mu*(grads))
                # Project

                lambda_dual = lambda_dual/torch.sum(lambda_dual).item()
                # lambda_dual_projected = toyexample.projsplx(lambda_dual.cpu().detach().numpy())
                # lambda_dual = torch.tensor(lambda_dual_projected).to(device)
                # lambda_dual = lambda_dual/torch.sum(lambda_dual).item()

                # print('norm lambda',torch.sum(lambda_dual))
                #
                # lambda_dual = 100*laplacian.projsplx(lambda_dual.cpu()).to(device)
                # print('norm lambda',torch.sum(lambda_dual))
            with torch.no_grad():
                acc = accuracy(net,test_loader,'cuda')
            loss_train = accuracy(net,train_loader,'cuda')
            utils.save_state(args.output_dir,epoch,loss_train,loss_train-loss_MSE,loss_MSE,acc)
            if epoch % 100 ==0 :
                print(epoch,loss.item(),'mu=',mu_dual.item(),loss.item()-loss_MSE,loss_train,acc)

    # net = net.to(device)    
    # for imgs, labels,_ in train_loader:
    #     imgs, labels = imgs.to(device), labels.to(device)
    #     output = net(imgs).to(device)           
    # print('output', output)
    # print('labels', labels)
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Manifold Regularization with Synthetic Data')

    parser.add_argument('--output_dir', type=str, default='smooth/out')
    parser.add_argument('--dataset', type=str, default='ground_robot_pavement', choices=['ground_robot_pavement','ground_robot_grass','quadrotor'])
    parser.add_argument('--data_dir', type=str, default='./smooth/data')


    parser.add_argument('--algorithm', type=str, default='ERM')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--regularizer', type=float, default=1)
    parser.add_argument('--heat_kernel_t', type=float, default=0.5)
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--clamp', type=float, default=0.3)
    
    parser.add_argument('--hidden_neurons', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bs', type=int, default=5)

    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.9)

    parser.add_argument('--dual_step_mu', type=float, default=0.5)
    parser.add_argument('--dual_step_lambda', type=float, default=0.1)
    parser.add_argument('--rho_step', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=0.01)

    args = parser.parse_args()
    args.output_dir = args.output_dir + '/' + str(args.dataset) +  '_' + args.algorithm+  '_'  + datetime.now().strftime("%Y-%m%d-%H%M%S")
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    main(args)