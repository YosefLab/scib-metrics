from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.utils import check_array

from .._types import IntOrKey
from ._dist import cdist, pdist_squareform
from ._utils import get_ndarray, validate_seed


def _initialize_random(X: jnp.ndarray, n_clusters: int, key: jax.random.KeyArray) -> jnp.ndarray:
    """Initialize cluster centroids randomly."""
    n_obs = X.shape[0]
    indices = jax.random.choice(key, n_obs, (n_clusters,), replace=False)
    initial_state = X[indices]
    return initial_state


@partial(jax.jit, static_argnums=1)
def _initialize_plus_plus(X: jnp.ndarray, n_clusters: int, key: jax.random.KeyArray) -> jnp.ndarray:
    """Initialize cluster centroids with k-means++ algorithm."""

    def _init(key):
        key, subkey = jax.random.split(key)
        # sample first centroid uniformly at random
        idx = jax.random.choice(subkey, n_obs)
        centroids = jnp.full((n_clusters,), -1, dtype=jnp.int32).at[0].set(idx)
        mask = jnp.zeros((n_obs,), dtype=jnp.bool_).at[idx].set(True)
        return centroids, mask, key

    def _step(state):
        centroids, mask, key = state
        key, subkey = jax.random.split(key)
        # d(x) = min_{mu in centroids} ||x - mu||^2, d(x) = 0 if x in centroids
        probs = jnp.where(mask, 0, jnp.min(jnp.where(mask, dists, jnp.inf), axis=1) ** 2)
        # sample with probability ~ d(x)
        idx = jax.random.choice(subkey, n_obs, p=probs / jnp.sum(probs))
        centroids = centroids.at[jnp.sum(mask)].set(idx)
        mask = mask.at[idx].set(True)
        return centroids, mask, key

    def _convergence(state):
        _, mask, _ = state
        return jnp.sum(mask) < n_clusters

    dists = pdist_squareform(X)
    n_obs = X.shape[0]
    centroids, _, _ = jax.lax.while_loop(_convergence, _step, _init(key))
    return X[centroids]


@jax.jit
def _get_dist_labels(X: jnp.ndarray, centroids: jnp.ndarray) -> jnp.ndarray:
    """Get the distance and labels for each observation."""
    dist = cdist(X, centroids)
    labels = jnp.argmin(dist, axis=1)
    return dist, labels


class KMeansJax:
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
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        seed: IntOrKey = 0,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.seed: jax.random.KeyArray = validate_seed(seed)

        if init not in ["k-means++", "random"]:
            raise ValueError("Invalid init method, must be one of ['k-means++' or 'random'].")
        if init == "k-means++":
            self._initialize = _initialize_plus_plus
        else:
            self._initialize = _initialize_random

    def fit(self, X: np.ndarray):
        """Fit the model to the data."""
        X = check_array(X, dtype=np.float32, order="C")
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
            old_inertia = state[1]
            centroids, _, _, n_iter = state
            # TODO(adamgayoso): Efficiently compute argmin and min simultaneously.
            dist, new_labels = _get_dist_labels(X, centroids)
            # From https://colab.research.google.com/drive/1AwS4haUx6swF82w3nXr6QKhajdF8aSvA?usp=sharing
            counts = (new_labels[jnp.newaxis, :] == jnp.arange(self.n_clusters)[:, jnp.newaxis]).sum(
                axis=1, keepdims=True
            )
            counts = jnp.clip(counts, a_min=1, a_max=None)
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
            new_inertia = jnp.mean(jnp.min(dist, axis=1))
            n_iter = n_iter + 1
            return new_centroids, new_inertia, old_inertia, n_iter

        def _kmeans_convergence(state):
            _, new_inertia, old_inertia, n_iter = state
            cond1 = jnp.abs(old_inertia - new_inertia) < self.tol
            cond2 = n_iter > self.max_iter
            return jnp.logical_or(cond1, cond2)[0]

        centroids = self._initialize(X, self.n_clusters, key)
        # centroids, new_inertia, old_inertia, n_iter
        state = (centroids, jnp.inf, jnp.inf, jnp.array([0.0]))
        state = _kmeans_step(state)
        state = jax.lax.while_loop(_kmeans_convergence, _kmeans_step, state)
        return state[0], state[1]
