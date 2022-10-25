import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def _euclidean_distance(x: np.array, y: np.array) -> float:
    dist = jnp.sqrt(jnp.sum((x - y) ** 2))
    return dist


@jax.jit
def cdist(x: np.ndarray, y: np.ndarray) -> jnp.ndarray:
    """Jax implementation of :func:`scipy.spatial.distance.cdist`.

    Uses euclidean distance.

    Parameters
    ----------
    x
        Array of shape (n_cells_a, n_features)
    y
        Array of shape (n_cells_b, n_features)

    Returns
    -------
    dist
        Array of shape (n_cells_a, n_cells_b)
    """
    return jax.vmap(lambda x1: jax.vmap(lambda y1: _euclidean_distance(x1, y1))(y))(x)


@jax.jit
def pdist_squareform(X: np.ndarray) -> jnp.ndarray:
    """Jax implementation of :func:`scipy.spatial.distance.pdist` and :func:`scipy.spatial.distance.squareform`.

    Uses euclidean distance.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features)

    Returns
    -------
    dist
        Array of shape (n_cells, n_cells)
    """
    n_cells = X.shape[0]
    inds = jnp.triu_indices(n_cells)

    def _body_fn(X, i_j):
        i, j = i_j
        return X, _euclidean_distance(X[i], X[j])

    dist_mat = jnp.zeros((n_cells, n_cells))
    dist_mat = dist_mat.at[inds].set(jax.lax.scan(_body_fn, X, (inds[0], inds[1]))[1])
    dist_mat = jnp.maximum(dist_mat, dist_mat.T)
    return dist_mat
