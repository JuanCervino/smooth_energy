

import torch
import torch.nn as nn
import os   
import argparse

import numpy as np

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

class FCNN(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim = 64, num_classes = 2):
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Manifold Regularization with Synthetic Data')

    parser.add_argument('--output_dir', type=str, default='toy')
    parser.add_argument('--dataset', type=str, default='two_moons')
    parser.add_argument('--n_dim', type=int, default=2, help='Dimension')
    parser.add_argument('--n_train', type=int, default=1)
    parser.add_argument('--n_unlab', type=int, default=100, help='Number of samples per class')
    parser.add_argument('--n_test', type=int, default=10)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--data_dir', type=str, default='./smooth/data')


    parser.add_argument('--algorithm', type=str, default='ERM')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--regularizer', type=float, default=1)
    parser.add_argument('--heat_kernel_t', type=float, default=0.05)
    parser.add_argument('--normalize', type=bool, default=True)

    parser.add_argument('--hidden_neurons', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
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

    # with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
    #     json.dump(args.__dict__, f, indent=2)

    # if args.dataset not in vars(datasets):
    #     raise NotImplementedError(f'Dataset {args.dataset} is not implemented.')

    main(args)