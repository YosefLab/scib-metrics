from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from sklearn.utils import check_array

from scib_metrics._types import IntOrKey

from ._dist import cdist
from ._utils import get_ndarray, validate_seed


def _tolerance(X: jnp.ndarray, tol: float) -> float:
    """Return a tolerance which is dependent on the dataset."""
    variances = np.var(X, axis=0)
    return np.mean(variances) * tol


def _initialize_random(X: jnp.ndarray, n_clusters: int, key: Array) -> jnp.ndarray:
    """Initialize cluster centroids randomly."""
    n_obs = X.shape[0]
    key, subkey = jax.random.split(key)
    indices = jax.random.choice(subkey, n_obs, (n_clusters,), replace=False)
    initial_state = X[indices]
    return initial_state


@partial(jax.jit, static_argnums=1)
def _initialize_plus_plus(X: jnp.ndarray, n_clusters: int, key: Array) -> jnp.ndarray:
    """Initialize cluster centroids with k-means++ algorithm."""
    n_obs = X.shape[0]
    key, subkey = jax.random.split(key)
    initial_centroid_idx = jax.random.choice(subkey, n_obs, (1,), replace=False)
    initial_centroid = X[initial_centroid_idx].ravel()
    dist_sq = jnp.square(cdist(initial_centroid[jnp.newaxis, :], X)).ravel()
    initial_state = {"min_dist_sq": dist_sq, "centroid": initial_centroid, "key": key}
    n_local_trials = 2 + int(np.log(n_clusters))

    def _step(state, _):
        prob = state["min_dist_sq"] / jnp.sum(state["min_dist_sq"])
        # note that observations already chosen as centers will have 0 probability
        # and will not be chosen again
        state["key"], subkey = jax.random.split(state["key"])
        next_centroid_idx_candidates = jax.random.choice(subkey, n_obs, (n_local_trials,), replace=False, p=prob)
        next_centroid_candidates = X[next_centroid_idx_candidates]
        # candidates by observations
        dist_sq_candidates = jnp.square(cdist(next_centroid_candidates, X))
        dist_sq_candidates = jnp.minimum(state["min_dist_sq"][jnp.newaxis, :], dist_sq_candidates)
        candidates_pot = dist_sq_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = jnp.argmin(candidates_pot)
        min_dist_sq = dist_sq_candidates[best_candidate]
        best_candidate = next_centroid_idx_candidates[best_candidate]

        state["min_dist_sq"] = min_dist_sq.ravel()
        state["centroid"] = X[best_candidate].ravel()
        return state, state["centroid"]

    _, centroids = jax.lax.scan(_step, initial_state, jnp.arange(n_clusters - 1))
    centroids = jnp.concatenate([initial_centroid[jnp.newaxis, :], centroids])
    return centroids


@jax.jit
def _get_dist_labels(X: jnp.ndarray, centroids: jnp.ndarray) -> jnp.ndarray:
    """Get the distance and labels for each observation."""
    dist = jnp.square(cdist(X, centroids))
    labels = jnp.argmin(dist, axis=1)
    return dist, labels


class KMeans:
    """Jax implementation of :class:`sklearn.cluster.KMeans`.

    This implementation is limited to Euclidean distance.

    Parameters
    ----------
    n_clusters
        Number of clusters.
    init
        Cluster centroid initialization method. One of the following:

        * ``'k-means++'``: Sample initial cluster centroids based on an
            empirical distribution of the points' contributions to the
            overall inertia.
        * ``'random'``: Uniformly sample observations as initial centroids
    n_init
        Number of times the k-means algorithm will be initialized.
    max_iter
        Maximum number of iterations of the k-means algorithm for a single run.
    tol
        Relative tolerance with regards to inertia to declare convergence.
    seed
        Random seed.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: Literal["k-means++", "random"] = "k-means++",
        n_init: int = 1,
        max_iter: int = 300,
        tol: float = 1e-4,
        seed: IntOrKey = 0,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol_scale = tol
        self.seed: jax.Array = validate_seed(seed)

        if init not in ["k-means++", "random"]:
            raise ValueError("Invalid init method, must be one of ['k-means++' or 'random'].")
        if init == "k-means++":
            self._initialize = _initialize_plus_plus
        else:
            self._initialize = _initialize_random

    def fit(self, X: np.ndarray):
        """Fit the model to the data."""
        X = check_array(X, dtype=np.float32, order="C")
        self.tol = _tolerance(X, self.tol_scale)
        # Subtract mean for numerical accuracy
        mean = X.mean(axis=0)
        X -= mean
        self._fit(X)
        X += mean
        self.cluster_centroids_ += mean
        return self

    def _fit(self, X: np.ndarray):
        all_centroids, all_inertias = jax.lax.map(
            lambda key: self._kmeans_full_run(X, key), jax.random.split(self.seed, self.n_init)
        )
        i = jnp.argmin(all_inertias)
        self.cluster_centroids_ = get_ndarray(all_centroids[i])
        self.inertia_ = get_ndarray(all_inertias[i])
        _, labels = _get_dist_labels(X, self.cluster_centroids_)
        self.labels_ = get_ndarray(labels)

    @partial(jax.jit, static_argnums=(0,))
    def _kmeans_full_run(self, X: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        def _kmeans_step(state):
            centroids, old_inertia, _, n_iter = state
            # TODO(adamgayoso): Efficiently compute argmin and min simultaneously.
            dist, new_labels = _get_dist_labels(X, centroids)
            # From https://colab.research.google.com/drive/1AwS4haUx6swF82w3nXr6QKhajdF8aSvA?usp=sharing
            counts = (new_labels[jnp.newaxis, :] == jnp.arange(self.n_clusters)[:, jnp.newaxis]).sum(
                axis=1, keepdims=True
            )
            counts = jnp.clip(counts, min=1, max=None)
            # Sum over points in a centroid by zeroing others out
            new_centroids = (
                jnp.sum(
                    jnp.where(
                        # axes: (data points, clusters, data dimension)
                        new_labels[:, jnp.newaxis, jnp.newaxis]
                        == jnp.arange(self.n_clusters)[jnp.newaxis, :, jnp.newaxis],
                        X[:, jnp.newaxis, :],
                        0.0,
                    ),
                    axis=0,
                )
                / counts
            )
            new_inertia = jnp.sum(jnp.min(dist, axis=1))
            n_iter = n_iter + 1
            return new_centroids, new_inertia, old_inertia, n_iter

        def _kmeans_convergence(state):
            _, new_inertia, old_inertia, n_iter = state
            cond1 = jnp.abs(old_inertia - new_inertia) > self.tol
            cond2 = n_iter < self.max_iter
            return jnp.logical_or(cond1, cond2)[0]

        centroids = self._initialize(X, self.n_clusters, key)
        # centroids, new_inertia, old_inertia, n_iter
        state = (centroids, jnp.inf, jnp.inf, jnp.array([0.0]))
        state = jax.lax.while_loop(_kmeans_convergence, _kmeans_step, state)
        # Compute final inertia
        centroids = state[0]
        dist, _ = _get_dist_labels(X, centroids)
        final_intertia = jnp.sum(jnp.min(dist, axis=1))
        return centroids, final_intertia
