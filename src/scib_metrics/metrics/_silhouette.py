from typing import Literal

import numpy as np
import pandas as pd

from scib_metrics.utils import silhouette_samples


def silhouette_label(
    X: np.ndarray,
    labels: np.ndarray,
    rescale: bool = True,
    chunk_size: int = 256,
    metric: Literal["euclidean", "cosine"] = "euclidean",
) -> float:
    """Average silhouette width (ASW) :cite:p:`luecken2022benchmarking`.

    Default parameters ('euclidean') match scIB implementation.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    labels
        Array of shape (n_cells,) representing label values
    rescale
        Scale asw into the range [0, 1].
    chunk_size
        Size of chunks to process at a time for distance computation
    metric
        The distance metric to use. The distance function can be 'euclidean' (default) or 'cosine'.

    Returns
    -------
    silhouette score
    """
    asw = np.mean(silhouette_samples(X, labels, chunk_size=chunk_size, metric=metric))
    if rescale:
        asw = (asw + 1) / 2
    return np.mean(asw)


def silhouette_batch(
    X: np.ndarray,
    labels: np.ndarray,
    batch: np.ndarray,
    rescale: bool = True,
    chunk_size: int = 256,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    between_cluster_distances: Literal["nearest", "mean_other", "furthest"] = "nearest",
) -> float:
    """Average silhouette width (ASW) with respect to batch ids within each label :cite:p:`luecken2022benchmarking`.

    Default parameters ('euclidean', 'nearest') match scIB implementation.

    Additional options enable BRAS compatible usage (see :func:`~scib_metrics.metrics.bras` documentation).

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    labels
        Array of shape (n_cells,) representing label values
    batch
        Array of shape (n_cells,) representing batch values
    rescale
        Scale asw into the range [0, 1]. If True, higher values are better.
    chunk_size
        Size of chunks to process at a time for distance computation.
    metric
        The distance metric to use. The distance function can be 'euclidean' (default) or 'cosine'.
    between_cluster_distances
        Method for computing inter-cluster distances.
        - 'nearest': Standard silhouette (distance to nearest cluster)
        - 'mean_other': BRAS-specific (mean distance to all other clusters)
        - 'furthest': BRAS-specific (distance to furthest cluster)

    Returns
    -------
    silhouette score
    """
    sil_dfs = []
    unique_labels = np.unique(labels)
    for group in unique_labels:
        labels_mask = labels == group
        X_subset = X[labels_mask]
        batch_subset = batch[labels_mask]
        n_batches = len(np.unique(batch_subset))

        if (n_batches == 1) or (n_batches == X_subset.shape[0]):
            continue

        sil_per_group = silhouette_samples(
            X_subset,
            batch_subset,
            chunk_size=chunk_size,
            metric=metric,
            between_cluster_distances=between_cluster_distances,
        )

        # take only absolute value
        sil_per_group = np.abs(sil_per_group)

        if rescale:
            # scale s.t. highest number is optimal
            sil_per_group = 1 - sil_per_group

        sil_dfs.append(
            pd.DataFrame(
                {
                    "group": [group] * len(sil_per_group),
                    "silhouette_score": sil_per_group,
                }
            )
        )

    sil_df = pd.concat(sil_dfs).reset_index(drop=True)
    sil_means = sil_df.groupby("group").mean()
    asw = sil_means["silhouette_score"].mean()

    return asw


def bras(
    X: np.ndarray,
    labels: np.ndarray,
    batch: np.ndarray,
    chunk_size: int = 256,
    metric: Literal["euclidean", "cosine"] = "cosine",
    between_cluster_distances: Literal["mean_other", "furthest"] = "mean_other",
) -> float:
    """Batch removal adapted silhouette (BRAS) for single-cell data integration assessment :cite:p:`rautenstrauch2025`.

    BRAS evaluates batch effect removal with respect to batch ids within each label (cell type cluster),
    using a modified silhouette score that accounts for nested batch effects. Unlike standard silhouette,
    BRAS computes between-cluster distances using the `between_cluster_distances` method rather than
    nearest-cluster approach. A higher scores indicates better batch mixing.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    labels
        Array of shape (n_cells,) representing label values
    batch
        Array of shape (n_cells,) representing batch values
    rescale
        Scale asw into the range [0, 1]. If True, higher values are better.
    chunk_size
        Size of chunks to process at a time for distance computation.
    metric
        The distance metric to use. The distance function can be 'euclidean' (default) or 'cosine'.
    between_cluster_distances
        Method for computing inter-cluster distances.
        - 'mean_other': Mean distance to all cells in other clusters (default)
        - 'furthest': Distance to furthest cluster (conservative estimate)

    Returns
    -------
    BRAS score
    """
    return silhouette_batch(X, labels, batch, True, chunk_size, metric, between_cluster_distances)
