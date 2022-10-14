from functools import partial
from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np

from ._dist import cdist, pdist_squareform

NdArray = Union[np.ndarray, jnp.ndarray]


@chex.dataclass
class _InterClusterData:
    inter_dist_a_b: jnp.ndarray
    inter_dist_b_a: jnp.ndarray
    inter_dist_per_label: jnp.ndarray
    indices_a: jnp.ndarray
    indices_b: jnp.ndarray


@jax.jit
def _intra_cluster_distances(X: jnp.ndarray):
    """Calculate the mean intra-cluster distance."""

    def _body_fn(cell_type_x):
        return _intra_cluster_distances_block(cell_type_x)

    # Labels by cells
    intra_dist_per_label = jax.lax.map(_body_fn, X)
    return intra_dist_per_label


@jax.jit
def _intra_cluster_distances_block(subset: jnp.ndarray) -> jnp.ndarray:
    mask = subset.sum(1) != 0
    full_mask = jnp.outer(mask, mask)
    distances = pdist_squareform(subset)
    per_cell_sum = jnp.where(full_mask, distances, 0).sum(axis=1)
    real_cells_in_subset = mask.sum()
    per_cell_mean = per_cell_sum / (real_cells_in_subset - 1)
    return per_cell_mean


# @jax.jit
def _nearest_cluster_distances(X: jnp.ndarray, inds: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the mean nearest-cluster distance for observation i."""

    def _body_fn(inds):
        i, j = inds
        return _nearest_cluster_distance_block(X[i], X[j])

    inter_dist = jax.lax.map(_body_fn, (inds[0], inds[1]))
    return inter_dist


@jax.jit
def _nearest_cluster_distance_block(subset_a: jnp.ndarray, subset_b: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mask_a = subset_a.sum(1) != 0
    mask_b = subset_b.sum(1) != 0
    full_mask = jnp.outer(mask_a, mask_b)
    distances = cdist(subset_a, subset_b)
    masked_distances = jnp.where(full_mask, distances, 0)
    values_a = masked_distances.sum(axis=1)
    values_b = masked_distances.sum(axis=0)
    real_cells_in_subset_a = mask_a.sum()
    real_cells_in_subset_b = mask_b.sum()
    values_a /= real_cells_in_subset_b
    values_b /= real_cells_in_subset_a
    return values_a, values_b


def _format_data(X: np.ndarray, labels: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reshape data to be labels by cells (padded) by features.

    The padding ensures each label has the same number of cells, which helps
    reduce the number of jit compilations that occur.
    """
    # TODO(adamgayoso): Make this jittable
    new_xs = []
    cumulative_mask = []
    _, largest_counts = np.unique(labels, return_counts=True)
    largest_label_count = np.max(largest_counts)
    original_inds = np.arange(X.shape[0])
    remapped_inds = []
    for l in np.unique(labels):
        subset_x = X[labels == l]
        new_x = np.zeros((largest_label_count, X.shape[1]))
        new_x[: subset_x.shape[0], :] = subset_x
        cumulative_mask += [True] * subset_x.shape[0] + [False] * (largest_label_count - subset_x.shape[0])
        new_xs.append(new_x)
        remapped_inds.append(original_inds[labels == l])
    cumulative_mask = jnp.array(cumulative_mask)
    remapped_inds = jnp.concatenate(remapped_inds)
    # labels by cells by features
    X = jnp.stack(new_xs)

    return X, cumulative_mask, remapped_inds


@partial(jax.jit, donate_argnums=(1,))
def _aggregate_inter_dists(i: int, inter_cluster_data: _InterClusterData) -> _InterClusterData:
    """Aggregate inter-cluster distances."""
    inter_dist_per_label = inter_cluster_data.inter_dist_per_label
    inter_dist_a_b = inter_cluster_data.inter_dist_a_b
    inter_dist_b_a = inter_cluster_data.inter_dist_b_a
    indices_a = inter_cluster_data.indices_a
    indices_b = inter_cluster_data.indices_b
    dist_a = inter_dist_per_label[indices_a[i]]
    dist_b = inter_dist_per_label[indices_b[i]]
    inter_dist_per_label = inter_dist_per_label.at[indices_a[i]].set(jnp.minimum(dist_a, inter_dist_a_b[i]))
    inter_dist_per_label = inter_dist_per_label.at[indices_b[i]].set(jnp.minimum(dist_b, inter_dist_b_a[i]))
    inter_cluster_data.inter_dist_per_label = inter_dist_per_label
    return inter_cluster_data


def silhouette_samples(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute the Silhouette Coefficient for each observation.

    Code inspired by:
    https://github.com/maxschelski/pytorch-cluster-metrics/

    Implements :func:`sklearn.metrics.silhouette_samples`.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features) representing a
        feature array.
    labels
        Array of shape (n_cells,) representing label values
        for each observation.

    Returns
    -------
    silhouette scores array of shape (n_cells,)
    """
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels should have the same number of samples")
    X, cumulative_mask, remapped_inds = _format_data(X, labels)
    n_labels = X.shape[0]

    # Compute intra-cluster distances
    intra_dist_shuffled = _intra_cluster_distances(X).ravel()[cumulative_mask]
    # Now unshuffle it
    intra_dist = jnp.zeros_like(intra_dist_shuffled)
    intra_dist = intra_dist.at[remapped_inds].set(intra_dist_shuffled)

    # Compute nearest-cluster distances
    inter_dist_per_label = jnp.inf * jnp.ones((n_labels, X.shape[1]))
    inter_inds = jnp.triu_indices(n_labels, k=1)
    inter_dist_a_b, inter_dist_b_a = _nearest_cluster_distances(X, inter_inds)
    del X
    inter_cluster_data = _InterClusterData(
        inter_dist_a_b=inter_dist_a_b,
        inter_dist_b_a=inter_dist_b_a,
        inter_dist_per_label=inter_dist_per_label,
        indices_a=inter_inds[0],
        indices_b=inter_inds[1],
    )
    # jax.lax.fori_loop is slow here
    for i in range(inter_inds[0].shape[0]):
        inter_cluster_data = _aggregate_inter_dists(i, inter_cluster_data)
    inter_dist_shuffled = inter_cluster_data.inter_dist_per_label.ravel()[cumulative_mask]
    inter_dist = jnp.zeros_like(inter_dist_shuffled)
    inter_dist = inter_dist.at[remapped_inds].set(inter_dist_shuffled)

    return np.array(jax.device_get((inter_dist - intra_dist) / jnp.maximum(intra_dist, inter_dist)))
