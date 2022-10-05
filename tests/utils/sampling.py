from typing import Union

import jax
import jax.numpy as jnp

IntOrKey = Union[int, jax.random.KeyArray]


def _validate_seed(seed: IntOrKey) -> jax.random.KeyArray:
    return jax.random.PRNGKey(seed) if isinstance(seed, int) else seed


def categorical_sample(
    n_obs: int,
    n_cats: int,
    seed: IntOrKey = 0,
) -> jnp.ndarray:
    return jax.random.categorical(_validate_seed(seed), jnp.ones(n_cats), shape=(n_obs,))


def uniform_sample(n_obs: int, n_vars: int, seed: IntOrKey) -> None:
    pass


def poisson_sample(
    n_obs: int,
    n_vars: int,
    rate: float = 1.0,
    seed: IntOrKey = 0,
) -> jnp.ndarray:
    return jax.random.poisson(_validate_seed(seed), rate, shape=(n_obs, n_vars))
