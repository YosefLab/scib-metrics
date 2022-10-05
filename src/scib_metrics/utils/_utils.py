from typing import Optional

import jax.numpy as jnp

from .._types import NdArray


def one_hot(y: NdArray, n_classes: Optional[int] = None) -> NdArray:
    """One-hot encode an array.

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
    y = jnp.resize(y, (y.shape[0]))
    n_vals = jnp.max(y) + 1
    # Ignore n_classes if smaller than inferred
    if n_classes and n_classes >= n_vals:
        n_vals = n_classes

    one_hot = jnp.eye(n_vals)[y]
    return one_hot.astype(jnp.uint8)
