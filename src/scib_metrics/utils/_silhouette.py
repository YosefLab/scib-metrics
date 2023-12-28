from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ._dist import cdist
from ._utils import get_ndarray


@jax.jit
def _silhouette_reduce(
    D_chunk: jnp.ndarray, labels_chunk: jnp.ndarray, labels: jnp.ndarray, label_freqs: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Accumulate silhouette statistics for vertical chunk of X.

    Follows scikit-learn implementation.

    Parameters
    ----------
    D_chunk
        Array of shape (n_chunk_samples, n_samples)
        Precomputed distances for a chunk.
    labels_chunk
        Array of shape (n_chunk_samples,)
        Labels for the chunk.
    start
        First index in the chunk.
    labels
        Array of shape (n_samples,)
        Corresponding cluster labels, encoded as {0, ..., n_clusters-1}.
    label_freqs
        Distribution of cluster labels in ``labels``.
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
    intra_index = (jnp.arange(D_chunk_len), labels_chunk)
    # intra_clust_dists are averaged over cluster size outside this function
    intra_clust_dists = clust_dists[intra_index]
    # of the remaining distances we normalise and extract the minimum
    clust_dists = clust_dists.at[intra_index].set(jnp.inf)
    clust_dists /= label_freqs
    inter_clust_dists = clust_dists.min(axis=1)
    return intra_clust_dists, inter_clust_dists


def _pairwise_distances_chunked(
    X: jnp.ndarray, chunk_size: int, reduce_fn: callable, labels: jnp.ndarray
) -> jnp.ndarray:
    """Compute pairwise distances in chunks to reduce memory usage."""
    n_samples = X.shape[0]
    n_chunks = jnp.ceil(n_samples / chunk_size).astype(int)
    # Pad data so the ragged last chunk does not trigger recompilation
    padded_obs = (chunk_size * n_chunks) - n_samples
    padded_data = jnp.concatenate([X, jnp.zeros((padded_obs, X.shape[-1]))], axis=0)
    padded_labels = jnp.concatenate([labels, jnp.zeros((padded_obs,), dtype=int)], axis=0)

    reshaped_padded_data = padded_data.reshape(n_chunks, chunk_size, -1)
    reshaped_padded_labels = padded_labels.reshape(n_chunks, chunk_size)

    map_fn = lambda chunk: reduce_fn(cdist(chunk[0], X), labels_chunk=chunk[1], labels=labels)
    out = jax.lax.map(map_fn, (reshaped_padded_data, reshaped_padded_labels))
    intra_dists_all_padded, inter_dists_all_padded = jax.tree_util.tree_map(
        lambda x: x.reshape(
            padded_data.shape[0],
        ),
        out,
    )

    # Now remove padded scores
    if padded_obs > 0:
        intra_dists_all = intra_dists_all_padded[:-padded_obs]
        inter_dists_all = inter_dists_all_padded[:-padded_obs]
    else:
        intra_dists_all = intra_dists_all_padded
        inter_dists_all = inter_dists_all_padded

    return intra_dists_all, inter_dists_all


def silhouette_samples(X: np.ndarray, labels: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    """Compute the Silhouette Coefficient for each observation.

    Implements :func:`sklearn.metrics.silhouette_samples`.

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

    Returns
    -------
    silhouette scores array of shape (n_cells,)
    """
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels should have the same number of samples")
    labels = pd.Categorical(labels).codes
    labels = jnp.asarray(labels)
    label_freqs = jnp.bincount(labels)
    reduce_fn = partial(_silhouette_reduce, label_freqs=label_freqs)
    results = _pairwise_distances_chunked(X, chunk_size=chunk_size, reduce_fn=reduce_fn, labels=labels)
    intra_clust_dists, inter_clust_dists = results

    denom = jnp.take(label_freqs - 1, labels, mode="clip")
    intra_clust_dists /= denom
    sil_samples = inter_clust_dists - intra_clust_dists
    sil_samples /= jnp.maximum(intra_clust_dists, inter_clust_dists)
    return get_ndarray(jnp.nan_to_num(sil_samples))
