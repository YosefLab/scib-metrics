import logging
import random
import warnings

import igraph
import numpy as np
from scipy.sparse import spmatrix
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.utils import check_array

from scib_metrics.nearest_neighbors import NeighborsResults
from scib_metrics.utils import KMeans

logger = logging.getLogger(__name__)


def _compute_clustering_kmeans(X: np.ndarray, n_clusters: int) -> np.ndarray:
    kmeans = KMeans(n_clusters)
    kmeans.fit(X)
    return kmeans.labels_


def _compute_clustering_leiden(connectivity_graph: spmatrix, resolution: float, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    igraph.set_random_number_generator(rng)
    # The connectivity graph with the umap method is symmetric, but we need to first make it directed
    # to have both sets of edges as is done in scanpy. See test for more details.
    g = igraph.Graph.Weighted_Adjacency(connectivity_graph, mode="directed")
    g.to_undirected(mode="each")
    clustering = g.community_leiden(objective_function="modularity", weights="weight", resolution=resolution)
    clusters = clustering.membership
    return np.asarray(clusters)


def _compute_nmi_ari_cluster_labels(
    X: spmatrix,
    labels: np.ndarray,
    resolution: float = 1.0,
    seed: int = 42,
) -> tuple[float, float]:
    labels_pred = _compute_clustering_leiden(X, resolution, seed)
    nmi = normalized_mutual_info_score(labels, labels_pred, average_method="arithmetic")
    ari = adjusted_rand_score(labels, labels_pred)
    return nmi, ari


def nmi_ari_cluster_labels_kmeans(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
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
    X: NeighborsResults,
    labels: np.ndarray,
    optimize_resolution: bool = True,
    resolution: float = 1.0,
    n_jobs: int = 1,
    seed: int = 42,
) -> dict[str, float]:
    """Compute nmi and ari between leiden clusters and labels.

    This deviates from the original implementation in scib by using leiden instead of
    louvain clustering. Installing joblib allows for parallelization of the leiden
    resoution optimization.

    Parameters
    ----------
    X
        A :class:`~scib_metrics.utils.nearest_neighbors.NeighborsResults` object.
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
    seed
        Seed used for reproducibility of clustering.

    Returns
    -------
    nmi
        Normalized mutual information score
    ari
        Adjusted rand index score
    """
    conn_graph = X.knn_graph_connectivities
    if optimize_resolution:
        n = 10
        resolutions = np.array([2 * x / n for x in range(1, n + 1)])
        try:
            from joblib import Parallel, delayed

            out = Parallel(n_jobs=n_jobs)(
                delayed(_compute_nmi_ari_cluster_labels)(conn_graph, labels, r) for r in resolutions
            )
        except ImportError:
            warnings.warn("Using for loop over clustering resolutions. `pip install joblib` for parallelization.")
            out = [_compute_nmi_ari_cluster_labels(conn_graph, labels, r, seed=seed) for r in resolutions]
        nmi_ari = np.array(out)
        nmi_ind = np.argmax(nmi_ari[:, 0])
        nmi, ari = nmi_ari[nmi_ind, :]
        return {"nmi": nmi, "ari": ari}
    else:
        nmi, ari = _compute_nmi_ari_cluster_labels(conn_graph, labels, resolution)

    return {"nmi": nmi, "ari": ari}
