from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnames=["metric"])
def cdist(x: np.ndarray, y: np.ndarray, metric: Literal["euclidean", "cosine"] = "euclidean") -> jnp.ndarray:
    """Jax implementation of :func:`scipy.spatial.distance.cdist`.

    Uses euclidean distance by default, cosine distance is also available.

    Parameters
    ----------
    x
        Array of shape (n_cells_a, n_features)
    y
        Array of shape (n_cells_b, n_features)
    metric
        The distance metric to use. The distance function can be 'euclidean' (default) or 'cosine'.

    Returns
    -------
    dist
        Array of shape (n_cells_a, n_cells_b)
    """
    if metric not in ["euclidean", "cosine"]:
        raise ValueError("Invalid metric choice, must be one of ['euclidean' or 'cosine'].")
    if metric == "cosine":
        # Normalize rows then use matmul — compiles to GEMM, no loop_reduce_fusion
        x_norm = x / jnp.linalg.norm(x, axis=1, keepdims=True)
        y_norm = y / jnp.linalg.norm(y, axis=1, keepdims=True)
        return jnp.clip(1.0 - x_norm @ y_norm.T, 0.0, 2.0)
    else:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b — compiles to GEMM, no loop_reduce_fusion
        sq_x = jnp.sum(x ** 2, axis=1, keepdims=True)
        sq_y = jnp.sum(y ** 2, axis=1)
        return jnp.sqrt(jnp.maximum(sq_x + sq_y - 2.0 * (x @ y.T), 0.0))


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
    # Use the same GEMM trick as cdist — avoids lax.scan loop_reduce_fusion on GPU
    sq = jnp.sum(X ** 2, axis=1)
    dist_mat = jnp.sqrt(jnp.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0))
    return dist_mat
