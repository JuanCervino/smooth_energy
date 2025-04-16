from datasets import load_dataset
import numpy as np
import torch
from smooth import laplacian
import matplotlib.pyplot as plt

def plot_matrix(matrix, title="Data Visualization"):
    plt.figure()
    # matrix = laplacian.get_pairwise_distance_matrix(data, 1, distance_type=title)
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()  # Add a colorbar to show the mapping of values to colors
    plt.title(title)
    pass

train_dataset = np.load("smooth_energy/data/inverted_pendulum_dataset.npz")

print("Loaded observations shape:", obs.shape)



obs_tensor = torch.tensor(train_dataset['obs'], dtype=torch.float32)
# act_tensor = torch.tensor(train_dataset['actions'], dtype=torch.float32)
act_tensor = torch.tensor(train_dataset['actions'], dtype=torch.float32).unsqueeze(1)

print(train_dataset)
print('total examples:', sum(train_dataset['episode_starts']))


print('Shape of train_dataset:', len(train_dataset['obs']))
# print('Shape of train_dataset:', len(train_dataset['actions'][0]))
obs_tensor_small = obs_tensor[90:150:5, :]  # Take only the first 10 observations for testing
act_tensor_small = act_tensor[90:150:5,:]  # Take only the first 10 observations for testing
print

matrix = {}

for distance_type in ['inverted_pendulum_batch', 
                      'euclidean',
                      'output']:
    print('distance_type:', distance_type)
    if distance_type == 'output':
        matrix[distance_type] = torch.cdist(act_tensor_small,act_tensor_small)
    else:
        matrix[distance_type] = laplacian.get_pairwise_distance_matrix(obs_tensor_small, 1, distance_type=distance_type)
    
    matrix[distance_type] = matrix[distance_type] / torch.max(matrix[distance_type])  # Normalize the matrix for better visualization
    plot_matrix(matrix[distance_type], title=f"Pairwise Distance Matrix ({distance_type})")

for distance_type in ['inverted_pendulum_batch', 
                      'euclidean']:

    plot_matrix(torch.abs(matrix[distance_type]-matrix['output']), title=f"Pairwise Difference Matrix ({distance_type})")
    print(distance_type, 'max:', torch.sum(torch.abs(matrix[distance_type])))

    print(distance_type, 'diff:', torch.sum(torch.abs(matrix[distance_type]-matrix['output'])))
print(distance_type, 'output:', torch.sum(torch.abs(matrix['output'])))

plt.show()