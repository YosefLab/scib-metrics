import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array

NdArray = np.ndarray | jnp.ndarray
IntOrKey = int | Array
ArrayLike = np.ndarray | sp.spmatrix | jnp.ndarray
