import torch
from scipy.stats import qmc
import numpy as np
import itertools

qmc_seed = 0

def test_generate_quadrature(N_interior, N_facet, dim):

    # Latin hypercube sampler
    latin_sampler = qmc.LatinHypercube(d=dim, scramble=True, seed=qmc_seed)
    # Sample interior points
    interior = latin_sampler.random(n=N_interior)
    # Index 0 indicates axis, index 1 indicates facet on that axis,
    # index 3 indicates quadrature point, index 4 indicates a component of that point.
    boundary = np.zeros((dim, 2, N_facet, dim))
    # Go through every axis
    for i in range(dim):
        # Go through every each facet
        for j in range(2):
            # Populate array
            boundary[i, j, :, :] = latin_sampler.random(n=N_facet)
        # Set x_i = 0
        boundary[i, 0, :, i] = 0
        # Set x_i = 1
        boundary[i, 1, :, i] = 1
    # Reshape into array containing all boundary points 
    boundary = boundary.reshape(2 * dim * N_facet, dim)
    
    # Return data
    return torch.tensor(interior, requires_grad=False), torch.tensor(boundary, requires_grad=False)


def generate_batch_quadrature(dim, total_batches=100, batch_size=400, split=.5):

    # Number of quadrature points in a batch that are in the interior
    batch_size_interior = int(batch_size * split)
    # Number of quadrature points in a batch that are on the boundary
    batch_size_boundary = batch_size - batch_size_interior
    # Number of quadrature points in a batch on any given facet of the boundary
    batch_size_facet = batch_size_boundary // (2 * dim)

    # Halton sampler
    halton_sampler = qmc.Halton(d=dim, scramble=True, seed=qmc_seed)

    # Index 0 indicates the batch, index 1 a quadrature point, index 2 a component of that point
    interior_batches = np.zeros((total_batches, batch_size_interior, dim))

    for i in range(total_batches):
        interior_batches[i, :, :] = halton_sampler.random(n=batch_size_interior)

    # Stack all the batches to obtain the entire set of quadrature points in the interior
    interior = interior_batches.reshape(total_batches * batch_size_interior, dim)

    boundary_batches = np.zeros((total_batches, batch_size_boundary, dim))

    # Go through every batch
    for i in range(total_batches):
        # Go through every axis
        for j in range(dim):
            # Quadrature points for x_j = 0
            sample_0 = halton_sampler.random(n=batch_size_facet)
            sample_0[:, j] = 0
            # Quadrature points for x_j = 1
            sample_1 = halton_sampler.random(n=batch_size_facet)
            sample_1[:, j] = 1
            # Stack the two samples
            sample = np.r_[sample_0, sample_1]
            # Add them to the current batch
            boundary_batches[i, 2*j*batch_size_facet:2*(j+1)*batch_size_facet, :] = sample

    # Stak all batches to obtain the entire set of quadrature points on the boundary
    boundary = boundary_batches.reshape(total_batches * batch_size_boundary, dim)

    # Convert to torch tensors
    interior, boundary, i_b, b_b = (torch.tensor(interior, requires_grad=False), torch.tensor(boundary, requires_grad=False),
        torch.tensor(interior_batches, requires_grad=False), torch.tensor(boundary_batches, requires_grad=False))

    # Return data
    return interior, boundary, itertools.cycle(i_b), itertools.cycle(b_b), batch_size_interior, batch_size_facet




