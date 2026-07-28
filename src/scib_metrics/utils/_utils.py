import warnings

import jax
import jax.numpy as jnp
import numpy as np
from chex import ArrayDevice
from jax import Array, nn
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array

from scib_metrics._types import ArrayLike, IntOrKey, NdArray


def get_ndarray(x: ArrayDevice) -> np.ndarray:
    """Convert Jax device array to Numpy array."""
    return np.array(jax.device_get(x))


def one_hot(y: NdArray, n_classes: int | None = None) -> jnp.ndarray:
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


def validate_seed(seed: IntOrKey) -> Array:
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
