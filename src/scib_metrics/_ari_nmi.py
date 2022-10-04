from typing import Tuple

import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

from .utils import KMeansJax


def _compute_clustering(X: np.ndarray, n_clusters: int) -> np.ndarray:
    kmeans = KMeansJax(n_clusters)
    kmeans.fit(X)
    return kmeans.labels_


def nmi_ari_cluster_labels(X: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Compute nmi and ari between k-means clusters and labels.

    This deviates from the original implementation in scib by using k-means
    with k equal to the known number of cell types/labels. This leads to
    a more efficient computation of the nmi and ari scores.

    Parameters
    ----------
    X
        Array of shape (n_samples, n_features).
    labels
        Array of shape (n_samples,) representing label values

    Returns
    -------
    nmi
        Normalized mutual information score
    ari
        Adjusted rand index score
    """
    n_clusters = len(np.unique(labels))
    labels_pred = _compute_clustering(X, n_clusters)
    nmi = normalized_mutual_info_score(labels, labels_pred, average_method="arithmetic")
    ari = adjusted_rand_score(labels, labels_pred)

    return nmi, ari
