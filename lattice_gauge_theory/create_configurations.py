#-------------------------------------------------------------------------------
# Filename: create_configurations.py

# Description: creates Monte Carlo configurations for the Ising lattice gauge
# theory. The output is then saved in the folder `configs/` as `train_labels.txt`,
# `train_configs.txt`, `test_labels.txt`, and `test_configs.txt`.
# For each temperature in `*_labels.txt`, there is a line of length N*N
# in `*_configs.txt`.

# Authors: Luca Papariello.

# Credits: Mark H. Fischer.
#-------------------------------------------------------------------------------

import numpy as np
# import matplotlib.pyplot as plt

# Size of the system NxN
N = 16
# Exchange coupling (ferromagnetic case)
J=1

# Number of low-T (T=0) configurations
N_low = 500
# Number of high-T (T=inf) configurations
N_high= 500


def initialize():
    '''
    Initializes a random spin configuration on a square lattice.

    Note: (i, j) denotes the plaquette to the right/up from vertex (i, j)
    and xy usually denotes the spin at +x/2 or +y/2 to the center.

    Returns:
        spins -- random spin configuration with format NxNx2, where 2
                 indicates the two sublattices.
    '''

    spins = 2*np.random.randint(2, size=((N, N, 2))) - np.ones((N, N, 2))
    return spins


def total_energy(spins):
    '''
    Get the total energy of the spin configuration.

    Arguments:
        spins -- spin configuration; shape: NxNx2; type: (int, int, 2).

    Returns:
        energy -- total energy of the spin configuration; type: float.
    '''

    N = np.shape(spins)[0]
    energy = 0

    for i in range(N):
        i_left = (i+N-1)%N
        for j in range(N):
            j_up = (N+j-1)%N
            en = -J * (spins[i, j, 0] * spins[i_left, j, 0] * spins[i, j, 1] * spins[i, j_up, 1])
            energy += en
    return energy


def dE(spins, i, j, xy):
    '''
    Calculates the energy difference of the current configuration compared to
    flipping the 'xy' spin at plaquette (i, j).

    Note:
    energy before flipping:
          for x spin:
            s(i,j)_x * [s(i-1,j)_x * s(i,j-1)_y * s(i,j)_y + s(i+1,j)_x * s(i+1,j-1)_y * s(i+1,j)_y]
          for y spin:
            s(i,j)_y * [s(i-1,j)_x * s(i,j-1)_y * s(i,j)_x + s(i-1,j+1)_x * s(i,j+1)_y * s(i,j+1)_x]

    energy after flipping: energy is -s(i,j)*[...] -> difference is 2*s(i,j)*[...].

    Arguments:
        spins -- spin configuration; shape: NxNx2; type: (int, int, 2).
        i, j -- plaquette at which to flip a spin, has to be in [0, N);
                type: int.
        xy -- which spin to flip {0, 1}; type: int.

    Returns:
        Energy difference after flipping the x or y spin at plaquette (i, j).
    '''

    i_right = (i+1)%N
    i_left = (i+N-1)%N
    j_down = (j+1)%N
    j_up = (N+j-1)%N

    if xy == 0:  # The x spin should be updated
        left_plaquette = spins[i, j, 0] * spins[i_left, j, 0] * spins[i, j_up, 1] * spins[i, j, 1]
        right_plaquette = spins[i, j, 0] * spins[i_right, j, 0] * spins[i_right, j_up, 1] * spins[i_right, j, 1]

        return 2*J*(left_plaquette+right_plaquette)
    else:  # Update the y spin
        up_plaquette = spins[i, j, 1] * spins[i_left, j, 0] * spins[i, j_up, 1] * spins[i, j, 0]
        down_plaquette = spins[i, j, 1] * spins[i_left, j_down, 0] * spins[i, j_down, 1] * spins[i, j_down, 0]

        return 2*J*(up_plaquette + down_plaquette)


def single_spin_update(spins, T):
    '''
    Performs a single step in a Metropolis single-spin update.

    Arguments:
        spins -- spin configuration; shape: NxNx2; type: (int, int, 2).
        T -- temperature for the probability; type: float.
    '''

    # First, choose the plaquette
    i, j = np.random.randint(N, size=2)
    # Then, choose whether to look at the x or y spin
    xy = np.random.randint(2)
    DE = dE(spins, i, j, xy)
    r = np.random.random()
    if T == 0 and DE <= 0:
        spins[i, j, xy] *= -1
        return
    if T == 0:
        return
    if r < np.exp(-DE/T):
        spins[i, j, xy] *= -1


def vertex_update(spins):
    '''
    Performs a vertex update, i.e., flipps all the spins around the vertex (i, j).
    Since this update does not change the energy, it is performed with probability 1.

    Arguments:
        spins -- spin configuration; shape: NxNx2; type: (int, int, 2).
    '''

    # Pick a vertex
    i, j = np.random.randint(N, size=2)
    i_left = (i+N-1)%N
    j_up = (N+j-1)%N
    # Flip every spin connected to it
    spins[i_left, j, 0] *= -1
    spins[i_left, j_up, 0] *= -1
    spins[i, j_up, 1] *= -1
    spins[i_left, j_up, 1] *= -1


Neq = 100000
Nupdate = N**2

configs = []
labels = []


# First create some zero-temperature configurations.
# This is could be done better/faster by starting
# from a fully polarized state and go from there.

spins = initialize()
not_yet = True
i=0

while not_yet:
    single_spin_update(spins, 0)
    vertex_update(spins)
    if i%100 == 0:
        if total_energy(spins) == -N**2:
            not_yet = False
    i+=1

print("create configurations for T=0")

for i in range(N_low*Nupdate):
    vertex_update(spins)
    if i%Nupdate == 0:
        configs.append(np.reshape(spins.copy(), N*N*2))
        labels.append(0)

print("configurations for T=0 done")


# Now for infinite-temperature configurations.
# Again, this could be much faster by creating fully random configurations.
# However, at least this allows for sampling any temperature...

spins = initialize()
for _ in range(Neq):
    single_spin_update(spins, np.inf)
    vertex_update(spins)

print("create configurations for T=inf")

for i in range(N_high*Nupdate):
    single_spin_update(spins, np.inf)
    vertex_update(spins)
    if i%Nupdate == 0:
        configs.append(np.reshape(spins.copy(), N*N*2))
        labels.append(1)

print("configurations for T=inf done")

# Training set
# np.savetxt("configs/train_configs.txt", configs, fmt='%i')
# np.savetxt("configs/train_labels.txt", labels, fmt='%i')

# Test set
np.savetxt("configs/test_configs.txt", configs, fmt='%i')
np.savetxt("configs/test_labels.txt", labels, fmt='%i')
