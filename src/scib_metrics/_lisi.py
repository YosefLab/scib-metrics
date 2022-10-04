from typing import Tuple
import numpy as np
from sklearn.utils import check_array
from scipy.sparse import csr_matrix, find

from scib_metrics.utils import compute_simpson_index


def _convert_knn_graph_to_idx(dist_mat: csr_matrix, knn_graph: csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    check_array(dist_mat, accept_sparse="csr")
    check_array(knn_graph, accept_sparse="csr")

    n_neighbors = np.array(knn_graph.sum(axis=1))
    if len(np.unique(n_neighbors)) > 1:
        raise ValueError("Each cell must have the same number of neighbors.")

    n_neighbors = int(np.unique(n_neighbors)[0])

    dist_mat_nonzero = find(dist_mat)

    nn_idx = np.empty(shape=(dist_mat.shape[0], n_neighbors), dtype=np.int32)
    nn_dists = np.empty(shape=(dist_mat.shape[0], n_neighbors), dtype=np.float32)
    nn_dists[:] = np.NaN
    for cell_id in np.arange(np.min(dist_mat_nonzero[0]), np.max(dist_mat_nonzero[0]) + 1):
        get_idx = dist_mat_nonzero[0] == cell_id
        num_idx = get_idx.sum()
        if num_idx < n_neighbors:
            raise ValueError(f"For cell {cell_id}, distance matrix did not contain distances for all neighbors.")
        nn_idx[cell_id, :n_neighbors] = dist_mat_nonzero[1][get_idx][
            np.argsort(dist_mat_nonzero[2][get_idx])
        ][:n_neighbors]
        nn_dists[cell_id, :n_neighbors] = np.sort(dist_mat_nonzero[2][get_idx])[:n_neighbors]
    return nn_dists, nn_idx


def lisi_knn(dist_mat: csr_matrix, knn_graph: csr_matrix, labels: np.ndarray, perplexity: float = None) -> np.ndarray:
    knn_dists, knn_idx = _convert_knn_graph_to_idx(dist_mat, knn_graph)

    if perplexity is None:
        perplexity = np.floor(knn_idx.shape[1] / 3)

    n_labels = len(np.unique(labels))

    simpson = compute_simpson_index(knn_dists, knn_idx, labels, n_labels, perplexity=perplexity)
    return 1 / simpson
