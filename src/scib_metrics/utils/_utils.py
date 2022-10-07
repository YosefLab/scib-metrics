from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import nn

from .._types import NdArray


def get_ndarray(x: jnp.ndarray) -> np.ndarray:
    """Convert Jax device array to Numpy array."""
    return np.array(jax.device_get(x))


def one_hot(y: NdArray, n_classes: Optional[int] = None) -> jnp.ndarray:
    """One-hot encode an array. Wrapper around :func:`~jax.nn.one_hot`.

    Parameters
    ----------
    y
        Array of shape (n_samples,) or (n_samples, 1).
    n_classes
        Number of classes. If None, inferred from the data.

    Returns
    -------
    one_hot: jnp.ndarray
        Array of shape (n_samples, n_classes).
    """
    n_classes = n_classes or jnp.max(y) + 1
    return nn.one_hot(jnp.ravel(y), n_classes)
