from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from chex import ArrayDevice
from jax import nn

from .._types import IntOrKey, NdArray


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
    n_classes = n_classes or jnp.max(y) + 1
    return nn.one_hot(jnp.ravel(y), n_classes)


def validate_seed(seed: IntOrKey) -> jax.random.KeyArray:
    """Validate a seed and return a Jax random key."""
    return jax.random.PRNGKey(seed) if isinstance(seed, int) else seed
