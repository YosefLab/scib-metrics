import anndata
import jax.numpy as jnp
import numpy as np
import pandas as pd
from harmonypy import compute_lisi as harmonypy_lisi
from scib.metrics import isolated_labels_asw
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist as sp_cdist
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans as SKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples as sk_silhouette_samples
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.neighbors import NearestNeighbors

import scib_metrics
from tests.utils.data import dummy_x_labels, dummy_x_labels_batch

scib_metrics.settings.jax_fix_no_kernel_image()


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
    X, labels, batches = dummy_x_labels_batch(x_is_neighbors_graph=True)
    scib_metrics.ilisi_knn(X, batches, perplexity=10)
    scib_metrics.clisi_knn(X, labels, perplexity=10)


def test_nmi_ari_cluster_labels_kmeans():
    X, labels = dummy_x_labels()
    out = scib_metrics.nmi_ari_cluster_labels_kmeans(X, labels)
    nmi, ari = out["nmi"], out["ari"]
    assert isinstance(nmi, float)
    assert isinstance(ari, float)


def test_nmi_ari_cluster_labels_leiden_parallel():
    X, labels = dummy_x_labels(return_symmetric_positive=True)
    out = scib_metrics.nmi_ari_cluster_labels_leiden(X, labels, optimize_resolution=True, n_jobs=2)
    nmi, ari = out["nmi"], out["ari"]
    assert isinstance(nmi, float)
    assert isinstance(ari, float)


def test_nmi_ari_cluster_labels_leiden_single_resolution():
    X, labels = dummy_x_labels(return_symmetric_positive=True)
    out = scib_metrics.nmi_ari_cluster_labels_leiden(X, labels, optimize_resolution=False, resolution=0.1)
    nmi, ari = out["nmi"], out["ari"]
    assert isinstance(nmi, float)
    assert isinstance(ari, float)


def test_isolated_labels():
    X, labels, batch = dummy_x_labels_batch()
    pred = scib_metrics.isolated_labels(X, labels, batch)
    adata = anndata.AnnData(X)
    adata.obsm["embed"] = X
    adata.obs["batch"] = batch
    adata.obs["labels"] = labels
    target = isolated_labels_asw(adata, "labels", "batch", "embed")
    np.testing.assert_allclose(np.array(pred), np.array(target))


def test_kmeans():
    centers = [[1, 1], [-1, -1], [1, -1]]
    len(centers)
    X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)
    kmeans = scib_metrics.utils.KMeans(n_clusters=3)
    kmeans.fit(X)
    assert kmeans.labels_.shape == (X.shape[0],)

    skmeans = SKMeans(n_clusters=3)
    skmeans.fit(X)
    sk_inertia = np.array([skmeans.inertia_])
    jax_inertia = np.array([kmeans.inertia_])
    np.testing.assert_allclose(sk_inertia, jax_inertia, atol=4e-2)

    # Reorder cluster centroids between methods and measure accuracy
    k_means_cluster_centers = k_means.cluster_centroids_
    order = pairwise_distances_argmin(k_means.cluster_centroids_, skmeans.cluster_centers_)
    sk_means_cluster_centers = skmeans.cluster_centers_[order]

    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
    sk_means_labels = pairwise_distances_argmin(X, sk_means_cluster_centers)

    accuracy = (k_means_labels == sk_means_labels).sum() / len(k_means_labels)
    assert accuracy > 0.999


def test_kbet():
    X, _, batch = dummy_x_labels_batch(x_is_neighbors_graph=True)
    acc_rate, stats, pvalues = scib_metrics.kbet(X, batch)
    assert isinstance(acc_rate, float)
    assert len(stats) == X.shape[0]
    assert len(pvalues) == X.shape[0]


def test_kbet_per_label():
    X, labels, batch = dummy_x_labels_batch(x_is_neighbors_graph=True)
    score = scib_metrics.kbet_per_label(X, batch, labels)
    assert isinstance(score, float)


def test_graph_connectivity():
    X, labels = dummy_x_labels(return_symmetric_positive=True)
    metric = scib_metrics.graph_connectivity(X, labels)
    assert isinstance(metric, float)
