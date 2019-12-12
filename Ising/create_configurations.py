#-------------------------------------------------------------------------------
# Filename: create_configurations.py.

# Description: creates Monte Carlo configurations for the Ising model. The
# output is then saved in the folder `configs/` as `train_labels_NxN.txt`,
# `train_configs_NxN.txt`, `test_labels_NxN.txt`, and `test_configs_NxN.txt`.
# For each temperature in `*_labels_NxN.txt`, there is a line of length N*N
# in `*_configs_NxN.txt`.

# Authors: Luca Papariello.

# Credits: Mark H. Fischer.
#-------------------------------------------------------------------------------

import numpy as np


# Size of the system NxN
N = 30
# Exchange coupling (ferromagnetic case)
J=1

# Number of cluster updates to thermalize
T_therm = 2000


def initialize():
    '''
    Initializes a random spin configuration on a square lattice.

    Returns:
        Random spin configuration with format NxN.
    '''

    return 2*np.random.randint(2, size=((N, N))) - np.ones((N, N))


def cluster_update(configuration, T):
    '''
    Performs a cluster update following the Wolff algorithm.

    Arguments:
        configuration -- spin configuration; shape: NxN; type: (int, int).
        T -- temperature for the probability; type: float.

    Returns:
        Size of cluster build and flipped.
    '''

    size = 0
    visited = np.zeros((N, N))
    cluster=[]
    # Choose random initial spin
    i, j = np.random.randint(N, size=2)
    cluster.append((i, j))
    visited[i, j]=1
    while len(cluster) > 0:
        i, j = cluster.pop()  #next i, j in line
        i_left = (i + 1)%N
        i_right = (i + N - 1)%N
        j_up = (j + 1)%N
        j_down = (N + j - 1)%N
        neighbors = [(i_left, j), (i_right, j), (i, j_up), (i, j_down)]
        for neighbor in neighbors:
            if visited[neighbor]==0 and configuration[neighbor] == configuration[i,j] and np.random.random() < (1-np.exp(-2*J/T)):
                cluster.append(neighbor)
                visited[neighbor] = 1
                size += 1
        configuration[i,j] *= -1
    return size


train_configs = []
train_labels = []

# How many temperatures between a min and max value.
num_T = 51
min_T = 1.0
max_T = 3.5

# How many configurations per temperature
num_conf = 50

# If we want to pick `num_T` random temperatures between `min_T` and `max_T`.
# Temps = min_T + np.random.random(num_T)*(max_T - min_T)
# Or if we want to pick `num_T` equally-spaced temperatures between `min_T` and `max_T`.
Temps = np.linspace(min_T, max_T, num_T)
for i, T in enumerate(Temps):
    print("create configurations for T=%.2f (%i / %i)" %(T, i+1, len(Temps)))
    configuration = initialize()
    csize = []
    # This is really an ad-hoc solution to the 'uncorrelated configurations'
    # problem, i.e. during some thermalization, the average cluster size is
    # calculated, then update roughly enough according to this size.
    for _ in range(T_therm):
        csize.append(cluster_update(configuration, T))
    T_A = int(N**2 / (2*np.mean(csize))) * 2 + 1
    for i in range(num_conf * T_A):
        cluster_update(configuration, T)
        if i%T_A == 0:
            train_configs.append(np.reshape(configuration.copy(), N**2))
            train_labels.append(T)

# Training set
np.savetxt("configs/train_labels_%ix%i.txt"%(N,N), train_labels, fmt='%.3f')
np.savetxt("configs/train_configs_%ix%i.txt"%(N,N), train_configs,  fmt='%i')

# Test set
# np.savetxt("configs/test_labels_%ix%i.txt"%(N,N), train_labels, fmt='%.3f')
# np.savetxt("configs/test_configs_%ix%i.txt"%(N,N), train_configs,  fmt='%i')
