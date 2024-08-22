import jax
import jax.numpy as jnp
from jax import Array

IntOrKey = int | Array


def _validate_seed(seed: IntOrKey) -> Array:
    return jax.random.PRNGKey(seed) if isinstance(seed, int) else seed


def categorical_sample(
    n_obs: int,
    n_cats: int,
    seed: IntOrKey = 0,
) -> jnp.ndarray:
    return jax.random.categorical(_validate_seed(seed), jnp.ones(n_cats), shape=(n_obs,))


def normal_sample(
    n_obs: int,
    mean: float = 0.0,
    var: float = 1.0,
    seed: IntOrKey = 0,
) -> jnp.ndarray:
    return jax.random.multivariate_normal(_validate_seed(seed), jnp.ones(n_obs) * mean, jnp.eye(n_obs) * var)


def poisson_sample(
    n_obs: int,
    n_vars: int,
    rate: float = 1.0,
    seed: IntOrKey = 0,
) -> jnp.ndarray:
    return jax.random.poisson(_validate_seed(seed), rate, shape=(n_obs, n_vars))
