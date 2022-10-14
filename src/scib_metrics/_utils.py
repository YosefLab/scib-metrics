from typing import Union

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

ArrayLike = Union[np.ndarray, sp.spmatrix, jnp.ndarray]


def _check_square(X: ArrayLike):
    if X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square matrix")
