import itertools
from dataclasses import dataclass
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np

from ._dist import cdist

NdArray = Union[np.ndarray, jnp.ndarray]


@dataclass
class _IntraClusterData:
    data: NdArray
    labels: NdArray
    unique_labels: NdArray
    intra_dist: NdArray


@dataclass
class _InterClusterData:
    data: NdArray
    labels: NdArray
    inter_dist: NdArray
    label_combos: NdArray


def _intra_cluster_distances(X: np.ndarray, labels: np.ndarray):
    """Calculate the mean intra-cluster distance."""
    intra_dist = jnp.zeros((X.shape[0],))
    unique_labels = jnp.unique(labels)
    # TODO(adamgayoso) See if we can lax this
    # output = jax.lax.fori_loop(0, len(unique_labels), _intra_cluster_distances_block, input)
    for i in range(len(unique_labels)):
        input = _IntraClusterData(X, labels, unique_labels, intra_dist)
        intra_dist = _intra_cluster_distances_block(i, input)
    return intra_dist


@jax.jit
def _intra_value(subset: np.ndarray) -> jnp.ndarray:
    distances = cdist(subset, subset)
    values = distances.sum(axis=1) / (distances.shape[0] - 1)
    return values


def _intra_cluster_distances_block(i: int, input: _IntraClusterData) -> jnp.ndarray:
    labels_inds = input.labels == input.unique_labels[i]
    subset = input.data[labels_inds]
    values = _intra_value(subset)
    intra_dist = input.intra_dist
    intra_dist = intra_dist.at[labels_inds].set(values)
    return intra_dist


def _nearest_cluster_distances(X: np.ndarray, labels: np.ndarray):
    """Calculate the mean nearest-cluster distance for observation i."""
    unique_labels = jnp.unique(labels)
    inter_dist = jnp.array(np.inf * np.ones((X.shape[0],)))
    label_combinations = jnp.array([list(i) for i in list(itertools.combinations(unique_labels, 2))])
    for i in range(len(label_combinations)):
        input = _InterClusterData(X, labels, inter_dist, label_combinations)
        inter_dist = _nearest_cluster_distance_block(i, input)
    return inter_dist


@jax.jit
def _inter_values(subset_a: np.ndarray, subset_b: np.ndarray) -> Union[jnp.ndarray, jnp.ndarray]:
    distances = cdist(subset_a, subset_b)
    values_a = distances.mean(axis=1)
    values_b = distances.mean(axis=0)
    return values_a, values_b


def _nearest_cluster_distance_block(inter_dist: jnp.ndarray, input: _InterClusterData) -> jnp.ndarray:
    label_a = input.label_combos[inter_dist, 0]
    label_b = input.label_combos[inter_dist, 1]
    label_mask_a = input.labels == label_a
    label_mask_b = input.labels == label_b
    subset_a = input.data[label_mask_a]
    subset_b = input.data[label_mask_b]
    dist_a, dist_b = _inter_values(subset_a, subset_b)
    inter_dist = input.inter_dist
    inter_dist = inter_dist.at[label_mask_a].set(jnp.minimum(dist_a, inter_dist[label_mask_a]))
    inter_dist = inter_dist.at[label_mask_b].set(jnp.minimum(dist_b, inter_dist[label_mask_b]))
    return inter_dist


def silhouette_samples(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute the Silhouette Coefficient for each observation.

    Code inspired by:
    https://github.com/maxschelski/pytorch-cluster-metrics/

    Parameters
    ----------
    X
        Array of shape (n_samples, n_features) representing a
        feature array.
    labels
        Array of shape (n_samples,) representing label values
        for each observation.

    Returns
    -------
    silhouette scores array of shape (n_samples,)
    """
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels should have the same number of samples")

    intra_dist = _intra_cluster_distances(X, labels)
    inter_dist = _nearest_cluster_distances(X, labels)
    return jax.device_get((inter_dist - intra_dist) / jnp.maximum(intra_dist, inter_dist))
