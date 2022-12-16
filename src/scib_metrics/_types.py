from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

NdArray = Union[np.ndarray, jnp.ndarray]
IntOrKey = Union[int, jax.random.KeyArray]
ArrayLike = Union[np.ndarray, sp.spmatrix, jnp.ndarray]
