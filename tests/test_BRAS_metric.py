from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples as sk_silhouette_samples
from sklearn.metrics.pairwise import pairwise_distances

import scib_metrics
from tests.utils.data import dummy_benchmarker_adata, dummy_x_labels


# v 0.5.3 with modifications for BRAS usage
def silhouette_batch_custom(
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

    Additional options enable BRAS compatible usage (see `bras()` documentation).

    This version uses a naive implementation for the silhouette score calculation, serving as a reference for the fast
    implementation provided in this package.

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

        sil_per_group = silhouette_samples_custom(
            X_subset,
            batch_subset,
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


def silhouette_samples_custom(X, cluster_labels, metric="euclidean", between_cluster_distances="nearest"):
    """

    Naive implementation of silhouette score modifications changing inter-cluster distance calculations for testing fast
     implementations.

    Experimental variants include:
    - Standard silhouette ('nearest' cluster distance)
    - BRAS-specific modifications ('mean_other', 'furthest')

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    cluster_labels
        Array of shape (n_cells,) representing cluster label values
    metric
        The distance metric to use. The distance function can be 'euclidean' (default) or 'cosine'.
    between_cluster_distances
        Method for computing inter-cluster distances.
        - 'nearest': Standard silhouette (distance to nearest cluster)
        - 'mean_other': BRAS-specific (mean distance to all other clusters)
        - 'furthest': BRAS-specific (distance to furthest cluster)

    Returns
    -------
    (Modified) silhouette scores with selected inter-cluster distance calcuation.
    """

    # Number of clusters
    unique_cluster_labels = np.unique(cluster_labels)
    n_clusters = len(unique_cluster_labels)

    # If there's only one cluster or no clusters, return 0 as silhouette score cannot be computed
    if n_clusters == 1 or n_clusters == 0:
        return 0

    # Initialize silhouette scores
    silhouette_scores = np.zeros(len(X))

    # Calculate pairwise distance matrix
    distance_matrix = pairwise_distances(X, metric=metric)

    for i in range(len(X)):
        # Points in the same cluster
        same_cluster = cluster_labels == cluster_labels[i]
        other_clusters = cluster_labels != cluster_labels[i]
        # Exclude the current point for intra-cluster distance
        same_cluster[i] = False

        # a: Mean distance from i to all other points in the same cluster
        if np.sum(same_cluster) == 0:
            silhouette_scores[i] = 0
            continue

        a = np.mean(distance_matrix[i, same_cluster])

        # b: Mean distance from i to all points in the furthest different cluster
        if between_cluster_distances == "furthest":
            b = np.max(
                [
                    np.mean(distance_matrix[i, cluster_labels == label])
                    for label in unique_cluster_labels
                    if label != cluster_labels[i]
                ]
            )

        # b: Mean distance from i to all points in any other cluster
        elif between_cluster_distances == "mean_other":
            b = np.mean(distance_matrix[i, other_clusters])

        # b: Mean distance from i to all points in the nearest different cluster
        else:
            b = np.min(
                [
                    np.mean(distance_matrix[i, cluster_labels == label])
                    for label in unique_cluster_labels
                    if label != cluster_labels[i]
                ]
            )

        # Silhouette score for point i
        silhouette_scores[i] = (b - a) / max(a, b)

    return silhouette_scores


def test_silhouette_samples_cosine():
    X, labels = dummy_x_labels()
    assert np.allclose(
        scib_metrics.utils.silhouette_samples(X, labels, metric="cosine"),
        silhouette_samples_custom(X, labels, metric="cosine"),
        atol=1e-5,
    )


def test_silhouette_samples_nearest():
    X, labels = dummy_x_labels()
    assert np.allclose(
        scib_metrics.utils.silhouette_samples(X, labels, between_cluster_distances="nearest"),
        silhouette_samples_custom(X, labels, between_cluster_distances="nearest"),
        atol=1e-5,
    )


def test_silhouette_samples_mean_other():
    X, labels = dummy_x_labels()
    assert np.allclose(
        scib_metrics.utils.silhouette_samples(X, labels, between_cluster_distances="mean_other"),
        silhouette_samples_custom(X, labels, between_cluster_distances="mean_other"),
        atol=1e-5,
    )


def test_silhouette_samples_furthest():
    X, labels = dummy_x_labels()
    assert np.allclose(
        scib_metrics.utils.silhouette_samples(X, labels, between_cluster_distances="furthest"),
        silhouette_samples_custom(X, labels, between_cluster_distances="furthest"),
        atol=1e-5,
    )


def test_silhouette_label():
    X, labels = dummy_x_labels()
    score = scib_metrics.silhouette_label(X, labels)
    score_sk = (np.mean(sk_silhouette_samples(X, labels)) + 1) / 2
    assert np.allclose(score, score_sk)


def test_silhouette_label_cosine():
    X, labels = dummy_x_labels()
    score = scib_metrics.silhouette_label(X, labels, metric="cosine")
    score_sk = (np.mean(sk_silhouette_samples(X, labels, metric="cosine")) + 1) / 2
    assert np.allclose(score, score_sk)


def test_bras():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    score = scib_metrics.bras(ad.obsm[emb_keys[0]], ad.obs[labels_key], ad.obs[batch_key])
    score_no = silhouette_batch_custom(
        ad.obsm[emb_keys[0]],
        ad.obs[labels_key],
        ad.obs[batch_key],
        metric="cosine",
        between_cluster_distances="mean_other",
    )
    assert np.allclose(score, score_no)


def test_silhouette_batch_default():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    score = scib_metrics.silhouette_batch(ad.obsm[emb_keys[0]], ad.obs[labels_key], ad.obs[batch_key])
    score_no = silhouette_batch_custom(ad.obsm[emb_keys[0]], ad.obs[labels_key], ad.obs[batch_key])
    assert np.allclose(score, score_no)


def test_silhouette_batch_cosine():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    score = scib_metrics.silhouette_batch(ad.obsm[emb_keys[0]], ad.obs[labels_key], ad.obs[batch_key], metric="cosine")
    score_no = silhouette_batch_custom(ad.obsm[emb_keys[0]], ad.obs[labels_key], ad.obs[batch_key], metric="cosine")
    assert np.allclose(score, score_no)


def test_silhouette_batch_nearest():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    score = scib_metrics.silhouette_batch(
        ad.obsm[emb_keys[0]], ad.obs[labels_key], ad.obs[batch_key], between_cluster_distances="nearest"
    )
    score_no = silhouette_batch_custom(
        ad.obsm[emb_keys[0]], ad.obs[labels_key], ad.obs[batch_key], between_cluster_distances="nearest"
    )
    assert np.allclose(score, score_no)


def test_silhouette_batch_cosine_nearest():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    score = scib_metrics.silhouette_batch(
        ad.obsm[emb_keys[0]],
        ad.obs[labels_key],
        ad.obs[batch_key],
        metric="cosine",
        between_cluster_distances="nearest",
    )
    score_no = silhouette_batch_custom(
        ad.obsm[emb_keys[0]],
        ad.obs[labels_key],
        ad.obs[batch_key],
        metric="cosine",
        between_cluster_distances="nearest",
    )
    assert np.allclose(score, score_no)


def test_silhouette_batch_furthest():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    score = scib_metrics.silhouette_batch(
        ad.obsm[emb_keys[0]], ad.obs[labels_key], ad.obs[batch_key], between_cluster_distances="furthest"
    )
    score_no = silhouette_batch_custom(
        ad.obsm[emb_keys[0]], ad.obs[labels_key], ad.obs[batch_key], between_cluster_distances="furthest"
    )
    assert np.allclose(score, score_no)


def test_silhouette_batch_cosine_furthest():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    score = scib_metrics.silhouette_batch(
        ad.obsm[emb_keys[0]],
        ad.obs[labels_key],
        ad.obs[batch_key],
        metric="cosine",
        between_cluster_distances="furthest",
    )
    score_no = silhouette_batch_custom(
        ad.obsm[emb_keys[0]],
        ad.obs[labels_key],
        ad.obs[batch_key],
        metric="cosine",
        between_cluster_distances="furthest",
    )
    assert np.allclose(score, score_no)


def test_silhouette_batch_mean_other():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    score = scib_metrics.silhouette_batch(
        ad.obsm[emb_keys[0]], ad.obs[labels_key], ad.obs[batch_key], between_cluster_distances="mean_other"
    )
    score_no = silhouette_batch_custom(
        ad.obsm[emb_keys[0]], ad.obs[labels_key], ad.obs[batch_key], between_cluster_distances="mean_other"
    )
    assert np.allclose(score, score_no)


def test_silhouette_batch_cosine_mean_other():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    score = scib_metrics.silhouette_batch(
        ad.obsm[emb_keys[0]],
        ad.obs[labels_key],
        ad.obs[batch_key],
        metric="cosine",
        between_cluster_distances="mean_other",
    )
    score_no = silhouette_batch_custom(
        ad.obsm[emb_keys[0]],
        ad.obs[labels_key],
        ad.obs[batch_key],
        metric="cosine",
        between_cluster_distances="mean_other",
    )
    assert np.allclose(score, score_no)
