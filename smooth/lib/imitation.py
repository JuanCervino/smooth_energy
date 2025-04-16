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
