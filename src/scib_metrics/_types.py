from typing import Union

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array

NdArray = Union[np.ndarray, jnp.ndarray]
IntOrKey = Union[int, Array]
ArrayLike = Union[np.ndarray, sp.spmatrix, jnp.ndarray]
