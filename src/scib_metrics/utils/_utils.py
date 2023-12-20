import warnings
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from chex import ArrayDevice
from jax import nn
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array

from scib_metrics._types import ArrayLike, IntOrKey, NdArray


def get_ndarray(x: ArrayDevice) -> np.ndarray:
    """Convert Jax device array to Numpy array."""
    return np.array(jax.device_get(x))


def one_hot(y: NdArray, n_classes: Optional[int] = None) -> jnp.ndarray:
    """One-hot encode an array. Wrapper around :func:`~jax.nn.one_hot`.

    Parameters
    ----------
    y
        Array of shape (n_cells,) or (n_cells, 1).
    n_classes
        Number of classes. If None, inferred from the data.

    Returns
    -------
    one_hot: jnp.ndarray
        Array of shape (n_cells, n_classes).
    """
    n_classes = n_classes or int(jax.device_get(jnp.max(y))) + 1
    return nn.one_hot(jnp.ravel(y), n_classes)


def validate_seed(seed: IntOrKey) -> jax.random.KeyArray:
    """Validate a seed and return a Jax random key."""
    return jax.random.PRNGKey(seed) if isinstance(seed, int) else seed


def check_square(X: ArrayLike):
    """Check if a matrix is square."""
    if X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square matrix")


def convert_knn_graph_to_idx(X: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Convert a kNN graph to indices and distances."""
    check_array(X, accept_sparse="csr")
    check_square(X)

    n_neighbors = np.unique(X.nonzero()[0], return_counts=True)[1]
    if len(np.unique(n_neighbors)) > 1:
        raise ValueError("Each cell must have the same number of neighbors.")

    n_neighbors = int(np.unique(n_neighbors)[0])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Precomputed sparse input")
        nn_obj = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed").fit(X)
        kneighbors = nn_obj.kneighbors(X)
    return kneighbors


def compute_connectivities_umap(
    knn_indices,
    knn_dists,
    n_obs,
    n_neighbors,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
):
    """Sped up version of sc.neighbors._compute_connectivities_umap."""
    with warnings.catch_warnings():
        # umap 0.5.0
        warnings.filterwarnings("ignore", message=r"Tensorflow not installed")
        from umap.umap_ import fuzzy_simplicial_set

    X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
    connectivities = fuzzy_simplicial_set(
        X,
        n_neighbors,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )

    if isinstance(connectivities, tuple):
        # In umap-learn 0.4, this returns (result, sigmas, rhos)
        connectivities = connectivities[0]

    n_samples = knn_indices.shape[0]
    distances = knn_dists.ravel()
    indices = knn_indices.ravel()

    # Check for self-connections
    self_connections = not np.all(knn_indices != np.arange(n_samples)[:, None])

    # Efficient creation of row pointer
    rowptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)

    # Create CSR matrix
    dist_sparse_csr = csr_matrix((distances, indices, rowptr), shape=(n_samples, n_samples))

    # Set diagonal to zero if self-connections exist
    if self_connections:
        dist_sparse_csr.setdiag(0.0)
    return dist_sparse_csr, connectivities.tocsr()
