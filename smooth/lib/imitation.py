import numpy as np
import torch
import os

def create_imitation_dataset(dataset, ids_train, ids_unlab, ids_test):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    assert dataset in ['inverted_pendulum']
    if dataset == 'inverted_pendulum':
        # Check if it exists
        file_name = 'smooth/data/inverted_pendulum_dataset.npz'
        # Load the dataset

    data = np.load(file_name)
    X_lab = torch.tensor(data['obs'][ids_train,:], dtype=torch.float32).to(device)
    y_lab = torch.tensor(data['actions'][ids_train], dtype=torch.float32).to(device).unsqueeze(1)
    X_unlab = torch.tensor(data['obs'][ids_unlab,:], dtype=torch.float32).to(device)
    y_unlab = torch.tensor(data['actions'][ids_unlab], dtype=torch.float32).to(device).unsqueeze(1)
    X_test = torch.tensor(data['obs'][ids_test,:], dtype=torch.float32).to(device)
    y_test = torch.tensor(data['actions'][ids_test], dtype=torch.float32).to(device).unsqueeze(1)
    return [X_lab, y_lab, X_unlab, y_unlab, X_test, y_test]


def create_imitation_dataset_trajectories(dataset, n_train, n_unlab, n_test, T):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    assert dataset in ['inverted_pendulum', 'swimmer', 'ant', 'halfcheetah', 'walker2d']
    if dataset == 'inverted_pendulum':
        # Check if it exists
        file_name = 'smooth/data/inverted_pendulum_dataset.npz'
        # Load the dataset
    elif dataset == 'swimmer':
        # Check if it exists
        file_name = 'smooth/data/swimmer_dataset.npz'
        # Load the dataset
    elif dataset == 'ant':
        # Check if it exists
        file_name = 'smooth/data/ant_dataset.npz'
        # Load the dataset
    elif dataset == 'halfcheetah':
        # Check if it exists
        file_name = 'smooth/data/half_cheetah_dataset.npz'
        # Load the dataset
    elif dataset == 'walker2d':
        # Check if it exists
        file_name = 'smooth/data/walker_dataset.npz'
        # Load the dataset
        
        
    data = np.load(file_name)
    start_indices = np.where(data['episode_starts'] == 1)[0]


    n_total = min(n_train + n_unlab + n_test, len(start_indices))

    ids_train = []
    ids_unlab = []
    ids_test = []

    for idx, start_idx in enumerate(start_indices[:n_total]):
        end_idx = start_indices[idx+1] if idx+1 < len(start_indices) else len(data['obs'])
        indices = list(range(start_idx, end_idx))
        indices = indices[:T]

        if idx < n_train:
            ids_train.extend(indices)
        elif idx < n_train + n_unlab:
            ids_unlab.extend(indices)
        else:
            ids_test.extend(indices)

    ids_train = np.array(ids_train)
    ids_unlab = np.array(ids_unlab)
    ids_test = np.array(ids_test)

    X_lab = torch.tensor(data['obs'][ids_train, :], dtype=torch.float32).to(device)
    y_lab = torch.tensor(data['actions'][ids_train], dtype=torch.float32).to(device)
    if len(y_lab.shape) == 1:
        y_lab = y_lab.unsqueeze(1)

    if len(ids_unlab) == 0:
        X_unlab = None
        y_unlab = None
    else:
        X_unlab = torch.tensor(data['obs'][ids_unlab, :], dtype=torch.float32).to(device)
        y_unlab = torch.tensor(data['actions'][ids_unlab], dtype=torch.float32).to(device)
        if len(y_lab.shape) == 1:
            y_unlab = y_unlab.unsqueeze(1)
            
    X_test = torch.tensor(data['obs'][ids_test, :], dtype=torch.float32).to(device)
    
    y_test = torch.tensor(data['actions'][ids_test], dtype=torch.float32).to(device)
    if len(y_test.shape) == 1:
        y_test = y_test.unsqueeze(1)
        
    return [X_lab, y_lab, X_unlab, y_unlab, X_test, y_test]