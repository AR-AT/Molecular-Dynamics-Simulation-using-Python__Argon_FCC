import numpy as np
from numba import njit, prange

EPSILON = 1.654017502e-21  
SIGMA = 3.405e-10  
MASS = 6.63e-26  
K_B = 1.38e-23 

def create_fcc_lattice(a, n):
    """Create an FCC lattice."""
    base = np.array([[0, 0, 0],
                     [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    positions = np.zeros((n**3 * 4, 3))
    index = 0
    for x in range(n):
        for y in range(n):
            for z in range(n):
                offset = np.array([x, y, z])
                for b in base:
                    positions[index] = (offset + b) * a
                    index += 1
    return positions

@njit
def lennard_jones(r2):
    """Calculate the Lennard-Jones potential and force."""
    r6 = (SIGMA ** 2 / r2) ** 3
    r12 = r6 ** 2
    potential = 4 * EPSILON * (r12 - r6)
    force = 24 * EPSILON * (2 * r12 - r6) / r2
    return potential

@njit(parallel=True)
def compute_forces(positions, box_length):
    """Compute forces using Lennard-Jones potential with PBC."""
    N = len(positions)
    potential_energy = 0.0

    for i in prange(N):
        for j in range(i + 1, N):
            r_vec = positions[i] - positions[j]
            r_vec = r_vec - box_length * np.round(r_vec / box_length)
            r2 = np.dot(r_vec, r_vec)
            if r2 < (2.5 * SIGMA) ** 2:
                potential = lennard_jones(r2)
                potential_energy += potential
                
    return potential_energy

# Create FCC lattice
lattice_parameter = 5.2742e-10
positions = create_fcc_lattice(lattice_parameter, 3)
box_length = 3 * lattice_parameter

# Calculate total potential energy
total_potential_energy = compute_forces(positions, box_length)

# Calculate average potential energy
average_potential_energy = total_potential_energy / 108

average_potential_energy
