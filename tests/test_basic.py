import sys

import jax.numpy as jnp
import numpy as np
import pandas as pd
from harmonypy import compute_lisi as harmonypy_lisi
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist as sp_cdist
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples as sk_silhouette_samples
from sklearn.neighbors import NearestNeighbors

import scib_metrics

scib_metrics.settings.jax_fix_no_kernel_image()

sys.path.append("../src/")


def dummy_x_labels(return_symmetric_positive=False):
    np.random.seed(1)
    X = np.random.normal(size=(100, 10))
    labels = np.random.randint(0, 2, size=(100,))
    if return_symmetric_positive:
        X = np.abs(X @ X.T)
    return X, labels


def dummy_x_labels_batch():
    X, labels = dummy_x_labels()
    batch = np.random.randint(0, 2, size=(100,))
    return X, labels, batch


def test_package_has_version():
    scib_metrics.__version__


def test_cdist():
    x = jnp.array([[1, 2], [3, 4]])
    y = jnp.array([[5, 6], [7, 8]])
    assert np.allclose(scib_metrics.utils.cdist(x, y), sp_cdist(x, y))


def test_pdist():
    x = jnp.array([[1, 2], [3, 4]])
    assert np.allclose(scib_metrics.utils.pdist_squareform(x), squareform(pdist(x)))


def test_silhouette_samples():
    X, labels = dummy_x_labels()
    assert np.allclose(scib_metrics.utils.silhouette_samples(X, labels), sk_silhouette_samples(X, labels), atol=1e-5)


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
    nbrs = NearestNeighbors(n_neighbors=30, algorithm="kd_tree").fit(X)
    D, knn_idx = nbrs.kneighbors(X)
    scib_metrics.utils.compute_simpson_index(
        jnp.array(D), jnp.array(knn_idx), jnp.array(labels), len(np.unique(labels))
    )


def test_lisi_knn():
    X, labels = dummy_x_labels()
    dist_mat = csr_matrix(scib_metrics.utils.cdist(X, X))
    nbrs = NearestNeighbors(n_neighbors=30, algorithm="kd_tree").fit(X)
    knn_graph = nbrs.kneighbors_graph(X)
    knn_graph = knn_graph.multiply(dist_mat)
    lisi_res = scib_metrics.lisi_knn(knn_graph, labels, perplexity=10)
    harmonypy_lisi_res = harmonypy_lisi(
        X, pd.DataFrame(labels, columns=["labels"]), label_colnames=["labels"], perplexity=10
    )[:, 0]
    assert np.allclose(lisi_res, harmonypy_lisi_res)


def test_ilisi_clisi_knn():
    X, labels, batches = dummy_x_labels_batch()
    dist_mat = csr_matrix(scib_metrics.utils.cdist(X, X))
    nbrs = NearestNeighbors(n_neighbors=30, algorithm="kd_tree").fit(X)
    knn_graph = nbrs.kneighbors_graph(X)
    knn_graph = knn_graph.multiply(dist_mat)
    scib_metrics.ilisi_knn(knn_graph, batches, perplexity=10)
    scib_metrics.clisi_knn(knn_graph, labels, perplexity=10)


def test_nmi_ari_cluster_labels_kmeans():
    X, labels = dummy_x_labels()
    nmi, ari = scib_metrics.nmi_ari_cluster_labels_kmeans(X, labels)
    assert isinstance(nmi, float)
    assert isinstance(ari, float)


def test_nmi_ari_cluster_labels_leiden_parallel():
    X, labels = dummy_x_labels(return_symmetric_positive=True)
    nmi, ari = scib_metrics.nmi_ari_cluster_labels_leiden(X, labels, optimize_resolution=True, n_jobs=2)
    assert isinstance(nmi, float)
    assert isinstance(ari, float)


def test_nmi_ari_cluster_labels_leiden_single_resolution():
    X, labels = dummy_x_labels(return_symmetric_positive=True)
    nmi, ari = scib_metrics.nmi_ari_cluster_labels_leiden(X, labels, optimize_resolution=False, resolution=0.1)
    assert isinstance(nmi, float)
    assert isinstance(ari, float)


def test_isolated_labels():
    X, labels, batch = dummy_x_labels_batch()
    scib_metrics.isolated_labels(X, labels, batch)


def test_kmeans():
    X, _ = dummy_x_labels()
    kmeans = scib_metrics.utils.KMeansJax(2)
    kmeans.fit(X)
    assert kmeans.labels_.shape == (X.shape[0],)
