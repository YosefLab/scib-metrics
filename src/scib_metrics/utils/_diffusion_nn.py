import logging
from typing import Literal

import numpy as np
import scipy
from scipy.sparse import csr_matrix, issparse

from scib_metrics import nearest_neighbors

logger = logging.getLogger(__name__)

_EPS = 1e-8


def _compute_transitions(X: csr_matrix, density_normalize: bool = True):
    """Code from scanpy.

    https://github.com/scverse/scanpy/blob/2e98705347ea484c36caa9ba10de1987b09081bf/scanpy/neighbors/__init__.py#L899
    """
    # TODO(adamgayoso): Refactor this with Jax
    # density normalization as of Coifman et al. (2005)
    # ensures that kernel matrix is independent of sampling density
    if density_normalize:
        # q[i] is an estimate for the sampling density at point i
        # it's also the degree of the underlying graph
        q = np.asarray(X.sum(axis=0))
        if not issparse(X):
            Q = np.diag(1.0 / q)
        else:
            Q = scipy.sparse.spdiags(1.0 / q, 0, X.shape[0], X.shape[0])
        K = Q @ X @ Q
    else:
        K = X

    # z[i] is the square root of the row sum of K
    z = np.sqrt(np.asarray(K.sum(axis=0)))
    if not issparse(K):
        Z = np.diag(1.0 / z)
    else:
        Z = scipy.sparse.spdiags(1.0 / z, 0, K.shape[0], K.shape[0])
    transitions_sym = Z @ K @ Z

    return transitions_sym


def _compute_eigen(
    transitions_sym: csr_matrix,
    n_comps: int = 15,
    sort: Literal["decrease", "increase"] = "decrease",
):
    """Compute eigen decomposition of transition matrix.

    https://github.com/scverse/scanpy/blob/2e98705347ea484c36caa9ba10de1987b09081bf/scanpy/neighbors/__init__.py
    """
    # TODO(adamgayoso): Refactor this with Jax
    matrix = transitions_sym
    # compute the spectrum
    if n_comps == 0:
        evals, evecs = scipy.linalg.eigh(matrix)
    else:
        n_comps = min(matrix.shape[0] - 1, n_comps)
        # ncv = max(2 * n_comps + 1, int(np.sqrt(matrix.shape[0])))
        ncv = None
        which = "LM" if sort == "decrease" else "SM"
        # it pays off to increase the stability with a bit more precision
        matrix = matrix.astype(np.float64)

        evals, evecs = scipy.sparse.linalg.eigsh(matrix, k=n_comps, which=which, ncv=ncv)
        evals, evecs = evals.astype(np.float32), evecs.astype(np.float32)
    if sort == "decrease":
        evals = evals[::-1]
        evecs = evecs[:, ::-1]

    return evals, evecs


def _get_sparse_matrix_from_indices_distances_numpy(indices, distances, n_obs, n_neighbors):
    """Code from scanpy."""
    n_nonzero = n_obs * n_neighbors
    indptr = np.arange(0, n_nonzero + 1, n_neighbors)
    D = csr_matrix(
        (
            distances.copy().ravel(),  # copy the data, otherwise strange behavior here
            indices.copy().ravel(),
            indptr,
        ),
        shape=(n_obs, n_obs),
    )
    D.eliminate_zeros()
    D.sort_indices()
    return D


def diffusion_nn(X: csr_matrix, k: int, n_comps: int = 100) -> nearest_neighbors.NeighborsResults:
    """Diffusion-based neighbors.

    This function generates a nearest neighbour list from a connectivities matrix.
    This allows us to select a consistent number of nearest neighbors across all methods.

    This differs from the original scIB implemenation by leveraging diffusion maps. Here we
    embed the data with diffusion maps in which euclidean distance represents well the diffusion
    distance. We then use pynndescent to find the nearest neighbours in this embedding space.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_cells) with non-zero values
        representing connectivities.
    k
        Number of nearest neighbours to select.
    n_comps
        Number of components for diffusion map

    Returns
    -------
    Neighbors results
    """
    transitions = _compute_transitions(X)
    evals, evecs = _compute_eigen(transitions, n_comps=n_comps)
    evals += _EPS  # Avoid division by zero
    # Multiscale such that the number of steps t gets "integrated out"
    embedding = evecs
    scaled_evals = np.array([e if e == 1 else e / (1 - e) for e in evals])
    embedding *= scaled_evals
    nn_result = nearest_neighbors.pynndescent(embedding, n_neighbors=k + 1)

    return nn_result
