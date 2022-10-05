from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array

from scib_metrics.utils import compute_simpson_index


def _convert_knn_graph_to_idx(knn_graph: csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    check_array(knn_graph, accept_sparse="csr")

    n_neighbors = np.unique(knn_graph.nonzero()[0], return_counts=True)[1]
    if len(np.unique(n_neighbors)) > 1:
        raise ValueError("Each cell must have the same number of neighbors.")

    n_neighbors = int(np.unique(n_neighbors)[0])

    nn_obj = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed").fit(knn_graph)
    return nn_obj.kneighbors(knn_graph)


def lisi_knn(knn_graph: csr_matrix, labels: np.ndarray, perplexity: float = None) -> np.ndarray:
    """Compute the local inverse simpson index (LISI) for each cell.

    Code inspired by:
    https://github.com/theislab/scib/blob/e578d84063adf4853ed087500bd3d67078e53337/scib/metrics/lisi.py#L586

    Parameters
    ----------
    knn_graph
        Sparse array of shape (n_samples, n_samples) with non-zero values for
        exactly each cell's k nearest neighbors.
    labels
        Array of shape (n_samples,) representing label values
        for each cell.
    perplexity
        Parameter controlling effective neighborhood size. If None, the
        perplexity is set to the number of neighbors // 3.
    """
    knn_dists, knn_idx = _convert_knn_graph_to_idx(knn_graph)

    if perplexity is None:
        perplexity = np.floor(knn_idx.shape[1] / 3)

    n_labels = len(np.unique(labels))

    simpson = compute_simpson_index(knn_dists, knn_idx, labels, n_labels, perplexity=perplexity)
    return 1 / simpson
