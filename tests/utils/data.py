import anndata
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import scib_metrics


def dummy_x_labels(return_symmetric_positive=False, x_is_neighbors_graph=False):
    np.random.seed(1)
    X = np.random.normal(size=(100, 10))
    labels = np.random.randint(0, 2, size=(100,))
    if return_symmetric_positive:
        X = np.abs(X @ X.T)
    if x_is_neighbors_graph:
        dist_mat = csr_matrix(scib_metrics.utils.cdist(X, X))
        nbrs = NearestNeighbors(n_neighbors=30, algorithm="kd_tree").fit(X)
        X = nbrs.kneighbors_graph(X)
        X = X.multiply(dist_mat)
    return X, labels


def dummy_x_labels_batch(x_is_neighbors_graph=False):
    X, labels = dummy_x_labels(x_is_neighbors_graph=x_is_neighbors_graph)
    batch = np.random.randint(0, 2, size=(100,))
    return X, labels, batch


def dummy_benchmarker_adata():
    X, labels, batch = dummy_x_labels_batch(x_is_neighbors_graph=False)
    adata = anndata.AnnData(X)
    labels_key = "labels"
    batch_key = "batch"
    adata.obs[labels_key] = labels
    adata.obs[batch_key] = batch
    embedding_keys = []
    for i in range(5):
        key = f"X_emb_{i}"
        adata.obsm[key] = X
        embedding_keys.append(key)
    return adata, embedding_keys, labels_key, batch_key
