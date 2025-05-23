import numpy as np
import torch

from torch_sparse import SparseTensor

# from sklearn.metrics import pairwise_distances
# import cvxpy as cp

#  From https://github.com/tegusi/RGCNN

def get_pairwise_euclidean_distance_matrix(tensor):
    """Compute pairwise distance of a tensor.
    Args:
        tensor: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    flat = tensor.flatten(start_dim = 1)
    adj_matrix = torch.cdist(flat,flat)

    return adj_matrix

def get_energy_between_points(p0, pT, T):
    """Compute energy between two points.
    Args:
        p0: tensor (num_dims)
        pT: tensor (num_dims)
        T: scalar
    Returns:
        energy: scalar
    """
    A = np.array([[144/5,144/4*1/2,48/3*1/2],
              [144/4*1/2,36/3,24/2*1/2],
              [48/3*1/2,24/2*1/2,4]])
    P = np.array([[T**5, T**4, T**3],
              [T**4, T**3, T**2],
              [T**3, T**2, T**1],])
    A = A * P 
    p = np.array([[T**4, T**3, T**2]
              ,[4*T**3, 3*T**2, 2*T]])
    b = np.array([  pT[1] - p0[0] - T * p0[1],
                    pT[1] - p0[1]
                    ])
    x = cp.Variable(3)

    prob = cp.Problem(cp.Minimize(cp.quad_form(x, A) ),
                    [p @ x == b])
    prob.solve()
    # print(p0.shape, pT.shape, prob.value)
    return prob.value

def get_energy_between_points_batch(p0, pT, T):
    """Compute energy between two sets of points in batch.
    Args:
        p0: tensor (N, num_dims) - First set of points
        pT: tensor (M, num_dims) - Second set of points
        T: scalar
    Returns:
        energy: tensor (N, M) - Pairwise energy between points
    """
    A = torch.tensor([[144/5, 144/4*1/2, 48/3*1/2],
                      [144/4*1/2, 36/3, 24/2*1/2],
                      [48/3*1/2, 24/2*1/2, 4]], device=p0.device)
    P = torch.tensor([[T**5, T**4, T**3],
                      [T**4, T**3, T**2],
                      [T**3, T**2, T**1]], device=p0.device)
    A = A * P

    p = torch.tensor([[T**4, T**3, T**2],
                      [4*T**3, 3*T**2, 2*T],
                      [0, 0, 1]], device=p0.device)  # Added a third row for consistency

    # Compute b for all pairs
    b = torch.stack([
        pT[:, 0] - p0[:, 0].unsqueeze(1) - T * p0[:, 1].unsqueeze(1),  # Shape: (N, M)
        pT[:, 1] - p0[:, 1].unsqueeze(1),                              # Shape: (N, M)
        torch.zeros_like(pT[:, 0]).unsqueeze(1).expand(-1, pT.size(0)) # Shape: (N, M)
    ], dim=2)  # Shape: (N, M, 3)

    # Expand A to match the pairwise dimensions of b
    A_expanded = A.unsqueeze(0).unsqueeze(0).expand(b.size(0), b.size(1), -1, -1)  # Shape: (N, M, 3, 3)

    # Solve the quadratic problem in batch
    x = torch.linalg.solve(A_expanded, b.unsqueeze(-1)).squeeze(-1)  # Shape: (N, M, 3)

    # Compute the quadratic form for energy
    energy = torch.einsum('ijk,ijkl,ijl->ij', x, A_expanded, x)  # Quadratic form

    return energy
def get_pairwise_distance_matrix(tensor, t, distance_type):
    """Compute pairwise distance of a tensor.
    Args:
        tensor: tensor (batch_size, num_points, num_dims)
        t: scalar
        distance_type: str, 'euclidean' or 'energy'
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    # t = 10.55 # Average distance of CIFAR10
    # t = 10.55**2 # Average distance square of CIFAR10
    if distance_type not in ['euclidean', 
                             'energy', 
                             'inverted_pendulum_batch', 
                             'inverted_pendulum_batch_vel_diff',
                             'momentum_pendulum', # This is not supported for InvertedPendulum-v4
                             ]:
        raise ValueError('distance_type should be euclidean or energy')
    
    if distance_type == 'euclidean':
        flat = tensor.flatten(start_dim = 1)
        adj_matrix = torch.cdist(flat,flat)
        adj_matrix = torch.square(adj_matrix)

        adj_matrix = torch.div(adj_matrix, -4*t)
        adj_matrix = torch.exp(adj_matrix)
        adj_matrix = adj_matrix.fill_diagonal_(0) # Delete the diagonal elements

    elif distance_type == 'energy':
        flat = tensor.flatten(start_dim=1)  # Shape: (N, D)

        # Extract x, y, z components for energy computation
        flat_x = flat[:, [0, 3]]  # Shape: (N, 2)
        flat_y = flat[:, [1, 4]]  # Shape: (N, 2)
        flat_z = flat[:, [2, 5]]  # Shape: (N, 2)

        # Compute pairwise energy distances for x, y, z components in batch
        energy_x = get_energy_between_points_batch(flat_x, flat_x, t)  # Shape: (N, N)
        energy_y = get_energy_between_points_batch(flat_y, flat_y, t)  # Shape: (N, N)
        energy_z = get_energy_between_points_batch(flat_z, flat_z, t)  # Shape: (N, N)

        # Combine energy components
        adj_matrix = energy_x + energy_y + energy_z

        # Ensure symmetry
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
    elif distance_type == 'inverted_pendulum_batch':
        """
        Calculate the total energy of the system for a batch of observations.
        Assumes the observation tensor is of shape [observations, 4], where each row is:
        [x_position, x_velocity, theta_angle, theta_velocity].
        """
        # Constants (example values, check MuJoCo XML for actual values)
        m_cart = 0.5  # Mass of the cart (example value, check MuJoCo XML)
        m_pendulum = 0.1  # Mass of the pendulum (example value)
        l = .3  # Half length of the pendulum (example value)
        g = 9.81  # Gravitational acceleration

        # Extract state components
        theta_angle = tensor[:, 1]  # Shape: [observations]
        x_velocity = tensor[:, 2]  # Shape: [observations]
        theta_velocity = tensor[:, 3]  # Shape: [observations]

        # Kinetic Energy
        KE_cart = 0.5 * m_cart * x_velocity**2  # Shape: [observations]
        v_pendulum = x_velocity + l * theta_velocity * torch.cos(theta_angle)  # Velocity of pendulum's center of mass
        KE_pendulum = 0.5 * m_pendulum * v_pendulum**2 + 0.5 * (1/3 * m_pendulum * l**2) * theta_velocity**2  # Shape: [observations]

        # Potential Energy
        h_pendulum = l * (1 - torch.cos(theta_angle))  # Height of pendulum's center of mass
        PE_pendulum = m_pendulum * g * h_pendulum  # Shape: [observations]

        # Total Energy
        total_energy = KE_cart + KE_pendulum + PE_pendulum  # Shape: [observations]
        # total_energy =   + PE_pendulum 
        
        total_energy = total_energy.unsqueeze(1)  # Reshape to (batch_size, 1) for pairwise distance calculation
        adj_matrix = torch.cdist(total_energy,total_energy, p=1)

    elif distance_type == 'inverted_pendulum_batch_vel_diff':
            """
            Calculate the total energy of the system for a batch of observations.
            Assumes the observation tensor is of shape [observations, 4], where each row is:
            [x_position, x_velocity, theta_angle, theta_velocity].
            """
            # Constants (example values, check MuJoCo XML for actual values)
            m_cart = 0.5  # Mass of the cart (example value, check MuJoCo XML)
            m_pendulum = 0.1  # Mass of the pendulum (example value)
            l = .3  # Half length of the pendulum (example value)
            g = 9.81  # Gravitational acceleration

            # Extract state components
            theta_angle = tensor[:, 1]  # Shape: [observations]
            x_velocity = tensor[:, 2]  # Shape: [observations]
            theta_velocity = tensor[:, 3]  # Shape: [observations]

            # Kinetic Energy
            x_velocity = x_velocity.unsqueeze(1)  # Reshape to (batch_size, 1) for pairwise distance calculation
            KE_cart = 0.5 * m_cart * torch.cdist(x_velocity,x_velocity)**2  # Shape: [observations]
            v_pendulum = x_velocity + l * theta_velocity * torch.cos(theta_angle)  # Velocity of pendulum's center of mass
            KE_pendulum = 0.5 * m_pendulum * torch.cdist(x_velocity,x_velocity)**2 + 0.5 * (1/3 * m_pendulum * l**2) * torch.cdist(x_velocity,x_velocity)**2  # Shape: [observations]

            # Potential Energy
            h_pendulum = l * (1 - torch.cos(theta_angle))  # Height of pendulum's center of mass
            h_pendulum = h_pendulum.unsqueeze(1)  # Reshape to (batch_size, 1) for pairwise distance calculation
            h_pendulum_repeat = h_pendulum.repeat(1, h_pendulum.size(0))  # Repeat for pairwise distance calculation
            PE_pendulum = m_pendulum * g * (h_pendulum_repeat-h_pendulum_repeat.T)  # Shape: [observations]

            # Total Energy
            adj_matrix = KE_cart + KE_pendulum + PE_pendulum  # Shape: [observations]
            # total_energy =   + PE_pendulum 
    elif distance_type == 'momentum_pendulum':
            """
            Calculate the momentum for a batch of observations.
            Assumes the observation tensor is of shape [observations, 4], where each row is:
            [x_position, x_velocity, theta_angle, theta_velocity].
            """
            m_cart = 0.5  # Mass of the cart (example value, check MuJoCo XML)
            m_pendulum = 0.1  # Mass of the pendulum (example value)
            l = .3  # Half length of the pendulum (example value)
            g = 9.81  # Gravitational acceleration

            # Extract state components
            theta_angle = tensor[:, 1].unsqueeze(1)  # Shape: [observations]
            x_velocity = tensor[:, 2].unsqueeze(1)  # Shape: [observations]
            theta_velocity = tensor[:, 3].unsqueeze(1)  # Shape: [observations]
            # Kinetic Energy
            # Calculate the momentum for the cart and pendulum
            # Momentum of the cart
            p_cart = (m_cart+m_pendulum) * x_velocity  # Shape: [observations, 1]
            p_pole = (l*m_pendulum/2) * theta_velocity * torch.cos(theta_angle)  # Shape: [observations, 1]
            adj_matrix = torch.cdist(p_cart, p_cart, p=1) + torch.cdist(p_pole, p_pole, p=1)  # Calculate pairwise momentum distances d
    return adj_matrix  # Shape: (batch_size, num_points, num_points)

def get_laplacian(imgs, normalize = False, heat_kernel_t = 10, clamp_value=None, distance_type = 'euclidean'):
    """Compute pairwise distance of a point cloud.

    Args:
        pairwise distance: tensor (batch_size, num_points, num_points)
        distance_type: str, 'euclidean' or 'energy'

    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if distance_type not in ['euclidean', 'energy','momentum_pendulum']:
        raise ValueError('distance_type should be euclidean or energy')
    
    
    adj_matrix = get_pairwise_distance_matrix(imgs, heat_kernel_t, distance_type)

    # Remove large values
    if clamp_value!=None:
        zero_tensor = torch.zeros(adj_matrix.size()).to(device)
        adj_matrix = torch.where(adj_matrix > clamp_value, adj_matrix, zero_tensor)

    if normalize:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        eye = torch.eye(adj_matrix.size()[0]).to(device) # Juan Modified This
        
        D = torch.diag(1 / torch.sqrt(D))
        L = eye - torch.matmul(torch.matmul(D, adj_matrix), D)
    else:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        # eye = tf.ones_like(D)
        # eye = tf.matrix_diag(eye)
        # D = 1 / tf.sqrt(D)
        D = torch.diag(D)
        L = D - adj_matrix
    return L

def get_laplacian_from_adj(adj_matrix, normalize = False, heat_kernel_t = 10, clamp_value=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Remove small values
    adj_matrix = torch.square(adj_matrix)

    adj_matrix = torch.div( adj_matrix, -4*heat_kernel_t)
    adj_matrix = torch.exp(adj_matrix)
    adj_matrix = adj_matrix.fill_diagonal_(0) # Delete the diagonal elements

    if clamp_value!=None:
        # remove large values
        zero_tensor = torch.zeros(adj_matrix.size()).to(device)
        adj_matrix = torch.where(adj_matrix < clamp_value, adj_matrix, zero_tensor)

    if normalize:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        eye = torch.eye(adj_matrix.size()[0]).to(device) # Juan Modified This
        D = torch.diag(1 / torch.sqrt(D))
        L = eye - torch.matmul(torch.matmul(D, adj_matrix), D)
    else:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        # eye = tf.ones_like(D)
        # eye = tf.matrix_diag(eye)
        # D = 1 / tf.sqrt(D)
        D = torch.diag(D)
        L = D - adj_matrix
    return L

def get_euclidean_laplacian_from_adj(adj_matrix, normalize = False, clamp_value=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Remove small values
    adj_matrix = torch.square(adj_matrix)

    # adj_matrix = torch.div( adj_matrix, -4*heat_kernel_t)
    # adj_matrix = torch.exp(adj_matrix)
    # adj_matrix = adj_matrix.fill_diagonal_(0) # Delete the diagonal elements

    if clamp_value!=None:
        zero_tensor = torch.zeros(adj_matrix.size()).to(device)
        adj_matrix = torch.where(adj_matrix < clamp_value, adj_matrix, zero_tensor)

    if normalize:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        eye = torch.eye(adj_matrix.size()[0]).to(device) # Juan Modified This
        D = torch.diag(1 / torch.sqrt(D))
        L = eye - torch.matmul(torch.matmul(D, adj_matrix), D)
    else:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        # eye = tf.ones_like(D)
        # eye = tf.matrix_diag(eye)
        # D = 1 / tf.sqrt(D)
        D = torch.diag(D)
        L = D - adj_matrix
    return L


def projsplx(tensor):
    hk1 = np.argsort(tensor)
    vals = tensor[hk1]
    n = len(vals)
    Flag = True
    i = n - 1
    while Flag:
        ti = (torch.sum(vals[i + 1:]) - 1) / (n - i)
        if ti >= vals[i]:
            Flag = False
            that = ti
        else:
            i = i - 1
        if i == 0:
            Flag = False
            that = (torch.sum(vals) - 1) / n
    vals = torch.nn.functional.relu(vals - that)
    vals = vals/torch.sum(vals).item()
    return vals[np.argsort(hk1)]


# def get_knn_matrix(tensor, distance_type = 'euclidean', matrix_type = 'knn', k=10):

#     if distance_type == 'euclidean':
#         flat = tensor.flatten(start_dim = 1)
#         adj_matrix = torch.cdist(flat,flat)
        

#     if matrix_type == 'knn':
#         assert isinstance(k, int), "k must be an integer"
#         rows, cols, weights = [], [], []

#         for i in range(adj_matrix.size(0)):
#             dists = adj_matrix[i].clone()
#             dists[i] = float('inf')  # Prevent self-loop
#             topk = torch.topk(-dists, k, largest=True)  # Negative for smallest distances
#             neighbors = topk.indices
#             weights_i = -topk.values

#             rows.append(torch.full((k,), i))
#             cols.append(neighbors)
#             weights.append(weights_i)

#         edge_index = torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)
#         edge_weight = torch.cat(weights)

#         adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=(adj_matrix.size(0), adj_matrix.size(0)))
#         return adj_t

# def get_knn_matrix(tensor, distance_type='euclidean', matrix_type='knn', k=10, batch_size=10):
#     assert matrix_type == 'knn', "Only 'knn' matrix_type is supported in this implementation."
#     assert distance_type == 'euclidean', "Only 'euclidean' distance_type is supported in this implementation."

#     flat = tensor.flatten(start_dim=1)
#     n_samples = flat.size(0)

#     rows, cols, weights = [], [], []

#     for start in range(0, n_samples, batch_size):
#         end = min(start + batch_size, n_samples)
#         batch = flat[start:end]  # Shape: (B, D)

#         # Compute pairwise distance between batch and all data
#         dists = torch.cdist(batch, flat)  # Shape: (B, N)
#         for i in range(end - start):
#             row_idx = start + i
#             dists_i = dists[i]
#             dists_i[row_idx] = float('inf')  # Avoid self-loop

#             topk = torch.topk(-dists_i, k, largest=True)
#             neighbors = topk.indices
#             weights_i = -topk.values

#             rows.append(torch.full((k,), row_idx, dtype=torch.long))
#             cols.append(neighbors)
#             weights.append(weights_i)

#     edge_index = torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)
#     edge_weight = torch.cat(weights)

#     adj_t = SparseTensor(
#         row=edge_index[0],
#         col=edge_index[1],
#         value=edge_weight,
#         sparse_sizes=(n_samples, n_samples)
#     )

#     return adj_t

def get_knn_matrix(tensor, distance_type='euclidean', matrix_type='knn', k=10, batch_size=10):
    assert matrix_type == 'knn', "Only 'knn' matrix_type is supported in this implementation."
    assert distance_type == 'euclidean', "Only 'euclidean' distance_type is supported in this implementation."

    device = tensor.device  # get device
    flat = tensor.flatten(start_dim=1)
    n_samples = flat.size(0)

    rows, cols, weights = [], [], []

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = flat[start:end]  # Shape: (B, D)

        dists = torch.cdist(batch, flat)  # Shape: (B, N)
        for i in range(end - start):
            row_idx = start + i
            dists_i = dists[i]
            dists_i[row_idx] = float('inf')  # Avoid self-loop

            topk = torch.topk(-dists_i, k, largest=True)
            neighbors = topk.indices
            weights_i = -topk.values

            rows.append(torch.full((k,), row_idx, dtype=torch.long, device=device))  # ensure correct device
            cols.append(neighbors.to(device))  # ensure correct device
            weights.append(weights_i.to(device))  # ensure correct device

    edge_index = torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)
    edge_weight = torch.cat(weights)

    adj_t = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=edge_weight,
        sparse_sizes=(n_samples, n_samples)
    )

    return adj_t


# def get_knn_matrix_efficient(tensor, distance_type='euclidean', matrix_type='knn', k=10, batch_size=512):
#     assert matrix_type == 'knn', "Only 'knn' matrix_type is supported in this implementation."
#     assert distance_type == 'euclidean', "Only 'euclidean' distance_type is supported in this implementation."

#     flat = tensor.flatten(start_dim=1)
#     N = flat.size(0)

#     # Store neighbors for each node
#     knn_indices = [[] for _ in range(N)]
#     knn_dists = [[] for _ in range(N)]

#     for i_start in range(0, N, batch_size):
#         i_end = min(i_start + batch_size, N)
#         x_i = flat[i_start:i_end]

#         for j_start in range(i_start, N, batch_size):
#             j_end = min(j_start + batch_size, N)
#             x_j = flat[j_start:j_end]

#             # Compute distance between x_i and x_j
#             dists = torch.cdist(x_i, x_j)  # Shape: (i_end - i_start, j_end - j_start)

#             for i in range(i_end - i_start):
#                 for j in range(j_end - j_start):
#                     global_i = i_start + i
#                     global_j = j_start + j

#                     if global_i == global_j:
#                         continue

#                     # Add distance to both (i, j) and (j, i) since symmetric
#                     knn_indices[global_i].append(global_j)
#                     knn_dists[global_i].append(dists[i, j])

#                     knn_indices[global_j].append(global_i)
#                     knn_dists[global_j].append(dists[i, j])

#     # Now select top-k neighbors for each node
#     rows, cols, weights = [], [], []

#     for i in range(N):
#         if len(knn_dists[i]) < k:
#             # Handle edge case: fewer than k neighbors
#             dists_i = torch.tensor(knn_dists[i])
#             neighbors_i = torch.tensor(knn_indices[i], dtype=torch.long)
#         else:
#             dists_i, indices = torch.topk(torch.tensor(knn_dists[i]), k=k, largest=False)
#             neighbors_i = torch.tensor(knn_indices[i], dtype=torch.long)[indices]

#         rows.append(torch.full((neighbors_i.size(0),), i, dtype=torch.long))
#         cols.append(neighbors_i)
#         weights.append(dists_i)

#     edge_index = torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)
#     edge_weight = torch.cat(weights)

#     adj_t = SparseTensor(
#         row=edge_index[0],
#         col=edge_index[1],
#         value=edge_weight,
#         sparse_sizes=(N, N)
#     )

#     return adj_t