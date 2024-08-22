from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np

from ._utils import get_ndarray

NdArray = np.ndarray | jnp.ndarray


@chex.dataclass
class _NeighborProbabilityState:
    H: float
    P: chex.ArrayDevice
    Hdiff: float
    beta: float
    betamin: float
    betamax: float
    tries: int


@jax.jit
def _Hbeta(knn_dists_row: jnp.ndarray, row_self_mask: jnp.ndarray, beta: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    P = jnp.exp(-knn_dists_row * beta)
    # Mask out self edges to be zero
    P = jnp.where(row_self_mask, P, 0)
    sumP = jnp.nansum(P)
    H = jnp.where(sumP == 0, 0, jnp.log(sumP) + beta * jnp.nansum(knn_dists_row * P) / sumP)
    P = jnp.where(sumP == 0, jnp.zeros_like(knn_dists_row), P / sumP)
    return H, P


@jax.jit
def _get_neighbor_probability(
    knn_dists_row: jnp.ndarray, row_self_mask: jnp.ndarray, perplexity: float, tol: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    beta = 1
    betamin = -jnp.inf
    betamax = jnp.inf
    H, P = _Hbeta(knn_dists_row, row_self_mask, beta)
    Hdiff = H - jnp.log(perplexity)

    def _get_neighbor_probability_step(state):
        Hdiff = state.Hdiff
        beta = state.beta
        betamin = state.betamin
        betamax = state.betamax
        tries = state.tries

        new_betamin = jnp.where(Hdiff > 0, beta, betamin)
        new_betamax = jnp.where(Hdiff > 0, betamax, beta)
        new_beta = jnp.where(
            Hdiff > 0,
            jnp.where(betamax == jnp.inf, beta * 2, (beta + betamax) / 2),
            jnp.where(betamin == -jnp.inf, beta / 2, (beta + betamin) / 2),
        )
        new_H, new_P = _Hbeta(knn_dists_row, row_self_mask, new_beta)
        new_Hdiff = new_H - jnp.log(perplexity)
        return _NeighborProbabilityState(
            H=new_H, P=new_P, Hdiff=new_Hdiff, beta=new_beta, betamin=new_betamin, betamax=new_betamax, tries=tries + 1
        )

    def _get_neighbor_probability_convergence(state):
        Hdiff, tries = state.Hdiff, state.tries
        return jnp.logical_and(jnp.abs(Hdiff) >= tol, tries < 50)

    init_state = _NeighborProbabilityState(H=H, P=P, Hdiff=Hdiff, beta=beta, betamin=betamin, betamax=betamax, tries=0)
    final_state = jax.lax.while_loop(_get_neighbor_probability_convergence, _get_neighbor_probability_step, init_state)
    return final_state.H, final_state.P


def _compute_simpson_index_cell(
    knn_dists_row: jnp.ndarray,
    knn_labels_row: jnp.ndarray,
    row_self_mask: jnp.ndarray,
    n_batches: int,
    perplexity: float,
    tol: float,
) -> jnp.ndarray:
    H, P = _get_neighbor_probability(knn_dists_row, row_self_mask, perplexity, tol)

    def _non_zero_H_simpson():
        sumP = jnp.bincount(knn_labels_row, weights=P, length=n_batches)
        return jnp.where(knn_labels_row.shape[0] == P.shape[0], jnp.dot(sumP, sumP), 1)

    return jnp.where(H == 0, -1, _non_zero_H_simpson())


def compute_simpson_index(
    knn_dists: NdArray,
    knn_idx: NdArray,
    row_idx: NdArray,
    labels: NdArray,
    n_labels: int,
    perplexity: float = 30,
    tol: float = 1e-5,
) -> np.ndarray:
    """Compute the Simpson index for each cell.

    Parameters
    ----------
    knn_dists
        KNN distances of size (n_cells, n_neighbors).
    knn_idx
        KNN indices of size (n_cells, n_neighbors) corresponding to distances.
    row_idx
        Idx of each row (n_cells, 1).
    labels
        Cell labels of size (n_cells,).
    n_labels
        Number of labels.
    perplexity
        Measure of the effective number of neighbors.
    tol
        Tolerance for binary search.

    Returns
    -------
    simpson_index
        Simpson index of size (n_cells,).
    """
    knn_dists = jnp.array(knn_dists)
    knn_idx = jnp.array(knn_idx)
    labels = jnp.array(labels)
    row_idx = jnp.array(row_idx)
    knn_labels = labels[knn_idx]
    self_mask = knn_idx != row_idx
    simpson_fn = partial(_compute_simpson_index_cell, n_batches=n_labels, perplexity=perplexity, tol=tol)
    out = jax.vmap(simpson_fn)(knn_dists, knn_labels, self_mask)
    return get_ndarray(out)
