from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ._dist import cdist
from ._utils import get_ndarray


@partial(jax.jit, static_argnames=["between_cluster_distances"])
def _silhouette_reduce(
    D_chunk: jnp.ndarray,
    start: int,
    labels: jnp.ndarray,
    label_freqs: jnp.ndarray,
    between_cluster_distances: Literal["nearest", "mean_other", "furthest"] = "nearest",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Accumulate silhouette statistics for vertical chunk of X.

    Follows scikit-learn implementation with default parameter usage ('nearest').

    Additional options enable BRAS compatible usage, addressing specific limitations of using silhouette in the context
     of evaluating data integration (see :func:`~scib_metrics.metrics.bras` documentation).


    Parameters
    ----------
    D_chunk
        Array of shape (n_chunk_samples, n_samples)
        Precomputed distances for a chunk.
    start
        First index in the chunk.
    labels
        Array of shape (n_samples,)
        Corresponding cluster labels, encoded as {0, ..., n_clusters-1}.
    label_freqs
        Distribution of cluster labels in ``labels``.
    between_cluster_distances
        Method for computing inter-cluster distances.
        - 'nearest': Standard silhouette (distance to nearest cluster)
        - 'mean_other': BRAS-specific (mean distance to all other clusters)
        - 'furthest': BRAS-specific (distance to furthest cluster)

    """
    # accumulate distances from each sample to each cluster
    D_chunk_len = D_chunk.shape[0]

    # If running into memory issues, use fori_loop instead of vmap
    # clust_dists = jnp.zeros((D_chunk_len, len(label_freqs)), dtype=D_chunk.dtype)
    # def _bincount(i, _data):
    #     clust_dists, D_chunk, labels, label_freqs = _data
    #     clust_dists = clust_dists.at[i].set(jnp.bincount(labels, weights=D_chunk[i], length=label_freqs.shape[0]))
    #     return clust_dists, D_chunk, labels, label_freqs

    # clust_dists = jax.lax.fori_loop(
    #     0, D_chunk_len, lambda i, _data: _bincount(i, _data), (clust_dists, D_chunk, labels, label_freqs)
    # )[0]
    clust_dists = jax.vmap(partial(jnp.bincount, length=label_freqs.shape[0]), in_axes=(None, 0))(labels, D_chunk)

    # intra_index selects intra-cluster distances within clust_dists
    intra_index = (jnp.arange(D_chunk_len), jax.lax.dynamic_slice(labels, (start,), (D_chunk_len,)))
    # intra_clust_dists are averaged over cluster size outside this function
    intra_clust_dists = clust_dists[intra_index]

    if between_cluster_distances == "furthest":
        # of the remaining distances we normalise and extract the maximum
        clust_dists = clust_dists.at[intra_index].set(-jnp.inf)
        clust_dists /= label_freqs
        inter_clust_dists = clust_dists.max(axis=1)
    elif between_cluster_distances == "mean_other":
        clust_dists = clust_dists.at[intra_index].set(jnp.nan)
        total_other_dists = jnp.nansum(clust_dists, axis=1)
        total_other_count = jnp.sum(label_freqs) - label_freqs[jax.lax.dynamic_slice(labels, (start,), (D_chunk_len,))]
        inter_clust_dists = total_other_dists / total_other_count
    elif between_cluster_distances == "nearest":
        # of the remaining distances we normalise and extract the minimum
        clust_dists = clust_dists.at[intra_index].set(jnp.inf)
        clust_dists /= label_freqs
        inter_clust_dists = clust_dists.min(axis=1)
    else:
        raise ValueError("Parameter 'between_cluster_distances' must be one of ['nearest', 'mean_other', 'furthest'].")
    return intra_clust_dists, inter_clust_dists


def _pairwise_distances_chunked(
    X: jnp.ndarray, chunk_size: int, reduce_fn: callable, metric: Literal["euclidean", "cosine"] = "euclidean"
) -> jnp.ndarray:
    """Compute pairwise distances in chunks to reduce memory usage."""
    n_samples = X.shape[0]
    n_chunks = jnp.ceil(n_samples / chunk_size).astype(int)
    intra_dists_all = []
    inter_dists_all = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_samples)
        intra_cluster_dists, inter_cluster_dists = reduce_fn(cdist(X[start:end], X, metric=metric), start=start)
        intra_dists_all.append(intra_cluster_dists)
        inter_dists_all.append(inter_cluster_dists)
    return jnp.concatenate(intra_dists_all), jnp.concatenate(inter_dists_all)


def silhouette_samples(
    X: np.ndarray,
    labels: np.ndarray,
    chunk_size: int = 256,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    between_cluster_distances: Literal["nearest", "mean_other", "furthest"] = "nearest",
) -> np.ndarray:
    """Compute the Silhouette Coefficient for each observation.

    Implements :func:`sklearn.metrics.silhouette_samples`.

    Default parameters ('euclidean', 'nearest') match scIB implementation.

    Additional options enable BRAS compatible usage (see `bras()` documentation).

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features) representing a
        feature array.
    labels
        Array of shape (n_cells,) representing label values
        for each observation.
    chunk_size
        Number of samples to process at a time for distance computation.
    metric
        The distance metric to use. The distance function can be 'euclidean' (default) or 'cosine'.
    between_cluster_distances
        Method for computing inter-cluster distances.
        - 'nearest': Standard silhouette (distance to nearest cluster)
        - 'mean_other': BRAS-specific (mean distance to all other clusters)
        - 'furthest': BRAS-specific (distance to furthest cluster)

    Returns
    -------
    silhouette scores array of shape (n_cells,)
    """
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels should have the same number of samples")
    labels = pd.Categorical(labels).codes
    labels = jnp.asarray(labels)
    label_freqs = jnp.bincount(labels)
    reduce_fn = partial(
        _silhouette_reduce, labels=labels, label_freqs=label_freqs, between_cluster_distances=between_cluster_distances
    )
    results = _pairwise_distances_chunked(X, chunk_size=chunk_size, reduce_fn=reduce_fn, metric=metric)
    intra_clust_dists, inter_clust_dists = results

    denom = jnp.take(label_freqs - 1, labels, mode="clip")
    intra_clust_dists /= denom
    sil_samples = inter_clust_dists - intra_clust_dists
    sil_samples /= jnp.maximum(intra_clust_dists, inter_clust_dists)
    return get_ndarray(jnp.nan_to_num(sil_samples))
