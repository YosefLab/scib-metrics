from typing import Union

import jax
import jax.numpy as jnp
import numpy as np

NdArray = Union[np.ndarray, jnp.ndarray]
IntOrKey = Union[int, jax.random.KeyArray]
