import logging
from typing import Dict, Tuple

import numpy as np
import scanpy as sc
from scipy.sparse import spmatrix
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.utils import check_array

from .utils import KMeansJax, check_square

logger = logging.getLogger(__name__)


def _compute_clustering_kmeans(X: np.ndarray, n_clusters: int) -> np.ndarray:
    kmeans = KMeansJax(n_clusters)
    kmeans.fit(X)
    return kmeans.labels_


def _compute_clustering_leiden(connectivity_graph: spmatrix, resolution: float) -> np.ndarray:
    g = sc._utils.get_igraph_from_adjacency(connectivity_graph)
    clustering = g.community_leiden(objective_function="modularity", weights="weight", resolution_parameter=resolution)
    clusters = clustering.membership
    return np.asarray(clusters)


def _compute_nmi_ari_cluster_labels(
    X: np.ndarray,
    labels: np.ndarray,
    resolution: float = 1.0,
) -> Tuple[float, float]:
    labels_pred = _compute_clustering_leiden(X, resolution)
    nmi = normalized_mutual_info_score(labels, labels_pred, average_method="arithmetic")
    ari = adjusted_rand_score(labels, labels_pred)
    return nmi, ari


def nmi_ari_cluster_labels_kmeans(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute nmi and ari between k-means clusters and labels.

    This deviates from the original implementation in scib by using k-means
    with k equal to the known number of cell types/labels. This leads to
    a more efficient computation of the nmi and ari scores.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    labels
        Array of shape (n_cells,) representing label values

    Returns
    -------
    nmi
        Normalized mutual information score
    ari
        Adjusted rand index score
    """
    X = check_array(X, accept_sparse=False, ensure_2d=True)
    n_clusters = len(np.unique(labels))
    labels_pred = _compute_clustering_kmeans(X, n_clusters)
    nmi = normalized_mutual_info_score(labels, labels_pred, average_method="arithmetic")
    ari = adjusted_rand_score(labels, labels_pred)

    return {"nmi": nmi, "ari": ari}


def nmi_ari_cluster_labels_leiden(
    X: spmatrix, labels: np.ndarray, optimize_resolution: bool = True, resolution: float = 1.0, n_jobs: int = 1
) -> Dict[str, float]:
    """Compute nmi and ari between leiden clusters and labels.

    This deviates from the original implementation in scib by using leiden instead of
    louvain clustering. Installing joblib allows for parallelization of the leiden
    resoution optimization.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_cells) representing a connectivity graph.
        Values should represent weights between pairs of neighbors, with a higher weight
        indicating more connected.
    labels
        Array of shape (n_cells,) representing label values
    optimize_resolution
        Whether to optimize the resolution parameter of leiden clustering by searching over
        10 values
    resolution
        Resolution parameter of leiden clustering. Only used if optimize_resolution is False.
    n_jobs
        Number of jobs for parallelizing resolution optimization via joblib. If -1, all CPUs
        are used.

    Returns
    -------
    nmi
        Normalized mutual information score
    ari
        Adjusted rand index score
    """
    X = check_array(X, accept_sparse=True, ensure_2d=True)
    check_square(X)
    if optimize_resolution:
        n = 10
        resolutions = np.array([2 * x / n for x in range(1, n + 1)])
        try:
            from joblib import Parallel, delayed

            out = Parallel(n_jobs=n_jobs)(delayed(_compute_nmi_ari_cluster_labels)(X, labels, r) for r in resolutions)
        except ImportError:
            logger.info("Using for loop over resolutions. pip install joblib for parallelization.")
            out = [_compute_nmi_ari_cluster_labels(X, labels, r) for r in resolutions]
        nmi_ari = np.array(out)
        nmi_ind = np.argmax(nmi_ari[:, 0])
        nmi, ari = nmi_ari[nmi_ind, :]
        return {"nmi": nmi, "ari": ari}
    else:
        nmi, ari = _compute_nmi_ari_cluster_labels(X, labels, resolution)

    return {"nmi": nmi, "ari": ari}
