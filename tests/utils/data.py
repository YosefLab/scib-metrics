import anndata
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import scib_metrics
from scib_metrics.nearest_neighbors import NeighborsResults


def dummy_x_labels(symmetric_positive=False, x_is_neighbors_results=False):
    rng = np.random.default_rng(seed=42)
    X = rng.normal(size=(100, 10))
    labels = rng.integers(0, 4, size=(100,))
    if symmetric_positive:
        X = np.abs(X @ X.T)
    if x_is_neighbors_results:
        dist_mat = csr_matrix(scib_metrics.utils.cdist(X, X))
        nbrs = NearestNeighbors(n_neighbors=30, metric="precomputed").fit(dist_mat)
        dist, ind = nbrs.kneighbors(dist_mat)
        X = NeighborsResults(indices=ind, distances=dist)
    return X, labels


def dummy_x_labels_batch(x_is_neighbors_results=False):
    rng = np.random.default_rng(seed=43)
    X, labels = dummy_x_labels(x_is_neighbors_results=x_is_neighbors_results)
    batch = rng.integers(0, 4, size=(100,))
    return X, labels, batch


def dummy_benchmarker_adata():
    X, labels, batch = dummy_x_labels_batch(x_is_neighbors_results=False)
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
