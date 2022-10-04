import itertools
from dataclasses import dataclass
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

NdArray = Union[np.ndarray, jnp.ndarray]


@jax.jit
def Hbeta(knn_dists_row: jnp.ndarray, beta: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Helper function for simpson index computation
    """
    P = jnp.exp(-knn_dists_row * beta)
    sumP = jnp.nansum(P)
    H = jnp.where(sumP == 0, 0, jnp.log(sumP) + beta * jnp.nansum(knn_dists_row * P) / sumP)
    P = jnp.where(sumP == 0, jnp.zeros_like(knn_dists_row), P / sumP)
    return H, P


@jax.jit
def _get_neighbor_probability(knn_dists_row: jnp.ndarray, perplexity: float, tol: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    beta = 1
    betamin = -jnp.inf
    betamax = jnp.inf
    H, P = Hbeta(knn_dists_row, beta)
    Hdiff = H - jnp.log(perplexity)

    def _get_neighbor_probability_step(state):
        _, _, Hdiff, beta, betamin, betamax, tries = state
        new_betamin = jnp.where(Hdiff > 0, beta, betamin)
        new_betamax = jnp.where(Hdiff > 0, betamax, beta)
        new_beta = jnp.where(Hdiff > 0, jnp.where(betamax == jnp.inf, beta * 2, (beta + betamax) / 2),
                             jnp.where(betamin == -jnp.inf, beta / 2, (beta + betamin) / 2))
        new_H, new_P = Hbeta(knn_dists_row, beta)
        new_Hdiff = new_H - jnp.log(perplexity)
        return new_H, new_P, new_Hdiff, new_beta, new_betamin, new_betamax, tries + 1

    def _get_neighbor_probability_convergence(state):
        _, _, Hdiff, _, _, _, tries = state
        return jnp.logical_and(jnp.abs(Hdiff) > tol, tries < 50)

    init_state = (H, P, Hdiff, beta, betamin, betamax, 0)
    final_state = jax.lax.while_loop(_get_neighbor_probability_convergence, _get_neighbor_probability_step, init_state)
    H, P, _, _, _, _, _ = final_state
    return H, P


def _compute_simpson_index_cell(
    knn_dists_row: jnp.ndarray, knn_row: jnp.ndarray, labels: jnp.ndarray, n_batches: int, perplexity: float, tol: float
) -> jnp.ndarray:
    H, P = _get_neighbor_probability(knn_dists_row, perplexity, tol)

    def _non_zero_H_simpson():
        knn_labels = jnp.take(labels, knn_row)
        L = jax.nn.one_hot(knn_labels, n_batches)
        sumP = P @ L
        return jnp.where(knn_labels.shape[0] == P.shape[0], jnp.dot(sumP, sumP), 1)

    return jnp.where(H == 0, -1, _non_zero_H_simpson())


def compute_simpson_index(
    knn_dists: NdArray,
    knn_idx: NdArray,
    labels: NdArray,
    n_labels: int,
    perplexity: float = 30,
    tol: float = 1e-5,
) -> np.ndarray:
    knn_dists = jnp.array(knn_dists)
    knn_idx = jnp.array(knn_idx)
    labels = jnp.array(labels)
    n = knn_dists.shape[0]
    return jax.device_get(jax.vmap(
        lambda i: _compute_simpson_index_cell(knn_dists[i, :], knn_idx[i, :], labels, n_labels, perplexity, tol)
    )(jnp.arange(n)))
