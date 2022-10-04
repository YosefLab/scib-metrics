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
        Array of shape (n_samples_a, n_features)
    y
        Array of shape (n_samples_b, n_features)

    Returns
    -------
    dist
        Array of shape (n_samples_a, n_samples_b)
    """
    return jax.vmap(lambda x1: jax.vmap(lambda y1: _euclidean_distance(x1, y1))(y))(x)
