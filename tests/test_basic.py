import scib_metrics
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist as sp_cdist
from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_samples as sk_silhouette_samples
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.append("../src/")


def dummy_x_labels():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    labels = np.array([0, 0, 1, 1, 0, 1])
    return X, labels


def dummy_x_labels_batch():
    X, labels = dummy_x_labels()
    batch = np.array([0, 1, 0, 1, 0, 1])
    return X, labels, batch


def test_package_has_version():
    scib_metrics.__version__


def test_cdist():
    x = jnp.array([[1, 2], [3, 4]])
    y = jnp.array([[5, 6], [7, 8]])
    assert np.allclose(scib_metrics.utils.cdist(x, y), sp_cdist(x, y))


def test_silhouette_samples():
    X, labels = dummy_x_labels()
    assert np.allclose(scib_metrics.utils.silhouette_samples(X, labels), sk_silhouette_samples(X, labels))


def test_silhouette_label():
    X, labels = dummy_x_labels()
    score = scib_metrics.silhouette_label(X, labels)
    assert score > 0
    scib_metrics.silhouette_label(X, labels, rescale=False)


def test_silhouette_batch():
    X, labels, batch = dummy_x_labels_batch()
    score = scib_metrics.silhouette_batch(X, labels, batch)
    assert score > 0
    scib_metrics.silhouette_batch(X, labels, batch)


def test_compute_simpson_index():
    X, labels = dummy_x_labels()
    D = scib_metrics.utils.cdist(X, X)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(X)
    D, knn_idx = nbrs.kneighbors(X)
    scib_metrics.utils.compute_simpson_index(jnp.array(D), jnp.array(
        knn_idx), jnp.array(labels), len(np.unique(labels)))

def test_lisi_knn():
    X, labels = dummy_x_labels()
    dist_mat = csr_matrix(scib_metrics.utils.cdist(X, X))
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(X)
    knn_graph = nbrs.kneighbors_graph(X)
    scib_metrics.lisi_knn(dist_mat, knn_graph, labels)


def test_isolated_labels():
    X, labels, batch = dummy_x_labels_batch()
    scib_metrics.isolated_labels(X, labels, batch)


def test_kmeans():
    X, _ = dummy_x_labels()
    kmeans = scib_metrics.utils.KMeansJax(2)
    kmeans.fit(X)
    assert kmeans.labels_.shape == (X.shape[0],)

