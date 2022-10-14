from typing import Union

import jax
import jax.numpy as jnp
import numpy as np

from ._dist import cdist, pdist_squareform

NdArray = Union[np.ndarray, jnp.ndarray]


@jax.jit
def _intra_cluster_distances(X: np.ndarray):
    """Calculate the mean intra-cluster distance."""
    # Labels by cells
    intra_dist_per_label = jax.vmap(_intra_cluster_distances_block, in_axes=0)(X)
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


@jax.jit
def _nearest_cluster_distances(X: np.ndarray):
    """Calculate the mean nearest-cluster distance for observation i."""
    inter_dist = jax.vmap(lambda x1: jax.vmap(lambda y1: _nearest_cluster_distance_block(x1, y1))(X))(X)
    return inter_dist


@jax.jit
def _nearest_cluster_distance_block(subset_a: np.ndarray, subset_b: np.ndarray) -> Union[jnp.ndarray, jnp.ndarray]:
    mask_a = subset_a.sum(1) != 0
    mask_b = subset_b.sum(1) != 0
    full_mask = jnp.outer(mask_a, mask_b)
    distances = cdist(subset_a, subset_b)
    masked_distances = jnp.where(full_mask, distances, 0)
    values_a = masked_distances.sum(axis=1)
    values_b = jnp.where(full_mask, distances, 0).sum(axis=0)
    real_cells_in_subset_a = mask_a.sum()
    real_cells_in_subset_b = mask_b.sum()
    values_a /= real_cells_in_subset_b
    values_b /= real_cells_in_subset_a
    return values_a, values_b


def silhouette_samples(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute the Silhouette Coefficient for each observation.

    Code inspired by:
    https://github.com/maxschelski/pytorch-cluster-metrics/

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
    new_xs = []
    cumulative_mask = []
    _, largest_counts = np.unique(labels, return_counts=True)
    largest_label_count = np.max(largest_counts)
    for l in np.unique(labels):
        subset_x = X[labels == l]
        new_x = np.zeros((largest_label_count, X.shape[1]))
        new_x[: subset_x.shape[0], :] = subset_x
        cumulative_mask += [True] * subset_x.shape[0] + [False] * (largest_label_count - subset_x.shape[0])
        new_xs.append(new_x)
    # labels by cells by features
    # cells dimension is same size for each label, padded with zeros
    # to make jit happy
    X = jnp.stack(new_xs)
    cumulative_mask = np.array(cumulative_mask)
    _intra_cluster_distances(X)
    inter_dist_a_b, inter_dist_b_a = _nearest_cluster_distances(X)
    # return jax.device_get((inter_dist - intra_dist) / jnp.maximum(intra_dist, inter_dist))
    # return intra_dist, inter_dist
