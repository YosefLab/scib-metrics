import anndata
import igraph
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from harmonypy import compute_lisi as harmonypy_lisi
from scib.metrics import isolated_labels_asw
from scipy.spatial.distance import cdist as sp_cdist
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans as SKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples as sk_silhouette_samples
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.neighbors import NearestNeighbors

import scib_metrics
from scib_metrics.nearest_neighbors import NeighborsResults
from tests.utils.data import dummy_x_labels, dummy_x_labels_batch

scib_metrics.settings.jax_fix_no_kernel_image()


def test_package_has_version():
    scib_metrics.__version__


def test_cdist():
    x = jnp.array([[1, 2], [3, 4]])
    y = jnp.array([[5, 6], [7, 8]])
    assert np.allclose(scib_metrics.utils.cdist(x, y), sp_cdist(x, y))


def test_cdist_cosine():
    x = jnp.array([[1, 2], [3, 4]])
    y = jnp.array([[5, 6], [7, 8]])
    assert np.allclose(scib_metrics.utils.cdist(x, y, metric="cosine"), sp_cdist(x, y, metric="cosine"), atol=1e-5)


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
    row_idx = np.arange(X.shape[0])[:, None]
    scib_metrics.utils.compute_simpson_index(
        jnp.array(D), jnp.array(knn_idx), jnp.array(row_idx), jnp.array(labels), len(np.unique(labels))
    )


@pytest.mark.parametrize("n_neighbors", [30, 60, 72])
def test_lisi_knn(n_neighbors):
    perplexity = n_neighbors // 3
    X, labels = dummy_x_labels()
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree").fit(X)
    dists, inds = nbrs.kneighbors(X)
    neigh_results = NeighborsResults(indices=inds, distances=dists)
    lisi_res = scib_metrics.lisi_knn(neigh_results, labels, perplexity=perplexity)
    harmonypy_lisi_res = harmonypy_lisi(
        X, pd.DataFrame(labels, columns=["labels"]), label_colnames=["labels"], perplexity=perplexity
    )[:, 0]
    np.testing.assert_allclose(lisi_res, harmonypy_lisi_res, rtol=5e-5, atol=5e-5)


def test_ilisi_clisi_knn():
    X, labels, batches = dummy_x_labels_batch(x_is_neighbors_results=True)
    scib_metrics.ilisi_knn(X, batches, perplexity=10)
    scib_metrics.clisi_knn(X, labels, perplexity=10)


def test_nmi_ari_cluster_labels_kmeans():
    X, labels = dummy_x_labels()
    out = scib_metrics.nmi_ari_cluster_labels_kmeans(X, labels)
    nmi, ari = out["nmi"], out["ari"]
    assert isinstance(nmi, float)
    assert isinstance(ari, float)


def test_nmi_ari_cluster_labels_leiden_parallel():
    X, labels = dummy_x_labels(symmetric_positive=True, x_is_neighbors_results=True)
    out = scib_metrics.nmi_ari_cluster_labels_leiden(X, labels, optimize_resolution=True, n_jobs=2)
    nmi, ari = out["nmi"], out["ari"]
    assert isinstance(nmi, float)
    assert isinstance(ari, float)


def test_nmi_ari_cluster_labels_leiden_single_resolution():
    X, labels = dummy_x_labels(symmetric_positive=True, x_is_neighbors_results=True)
    out = scib_metrics.nmi_ari_cluster_labels_leiden(X, labels, optimize_resolution=False, resolution=0.1)
    nmi, ari = out["nmi"], out["ari"]
    assert isinstance(nmi, float)
    assert isinstance(ari, float)


def test_nmi_ari_cluster_labels_leiden_reproducibility():
    X, labels = dummy_x_labels(symmetric_positive=True, x_is_neighbors_results=True)
    out1 = scib_metrics.nmi_ari_cluster_labels_leiden(X, labels, optimize_resolution=False, resolution=3.0)
    out2 = scib_metrics.nmi_ari_cluster_labels_leiden(X, labels, optimize_resolution=False, resolution=3.0)
    nmi1, ari1 = out1["nmi"], out1["ari"]
    nmi2, ari2 = out2["nmi"], out2["ari"]
    assert nmi1 == nmi2
    assert ari1 == ari2


def test_leiden_graph_construction():
    X, _ = dummy_x_labels(symmetric_positive=True, x_is_neighbors_results=True)
    conn_graph = X.knn_graph_connectivities
    g = igraph.Graph.Weighted_Adjacency(conn_graph, mode="directed")
    g.to_undirected(mode="each")
    sc_g = sc._utils.get_igraph_from_adjacency(conn_graph, directed=False)
    assert g.isomorphic(sc_g)
    np.testing.assert_equal(g.es["weight"], sc_g.es["weight"])


def test_isolated_labels():
    X, labels, batch = dummy_x_labels_batch()
    pred = scib_metrics.isolated_labels(X, labels, batch)
    adata = anndata.AnnData(X)
    adata.obsm["embed"] = X
    adata.obs["batch"] = batch
    adata.obs["labels"] = labels
    target = isolated_labels_asw(adata, "labels", "batch", "embed", iso_threshold=5)
    np.testing.assert_allclose(np.array(pred), np.array(target))


def test_kmeans():
    centers = np.array([[1, 1], [-1, -1], [1, -1]]) * 2
    len(centers)
    X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7, random_state=42)

    kmeans = scib_metrics.utils.KMeans(n_clusters=3)
    kmeans.fit(X)
    assert kmeans.labels_.shape == (X.shape[0],)

    skmeans = SKMeans(n_clusters=3)
    skmeans.fit(X)
    sk_inertia = np.array([skmeans.inertia_])
    jax_inertia = np.array([kmeans.inertia_])
    np.testing.assert_allclose(sk_inertia, jax_inertia, atol=4e-2)

    # Reorder cluster centroids between methods and measure accuracy
    k_means_cluster_centers = kmeans.cluster_centroids_
    order = pairwise_distances_argmin(kmeans.cluster_centroids_, skmeans.cluster_centers_)
    sk_means_cluster_centers = skmeans.cluster_centers_[order]

    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
    sk_means_labels = pairwise_distances_argmin(X, sk_means_cluster_centers)

    accuracy = (k_means_labels == sk_means_labels).sum() / len(k_means_labels)
    assert accuracy > 0.995


def test_kbet():
    X, _, batch = dummy_x_labels_batch(x_is_neighbors_results=True)
    acc_rate, stats, pvalues = scib_metrics.kbet(X, batch)
    assert isinstance(acc_rate, float)
    assert len(stats) == X.indices.shape[0]
    assert len(pvalues) == X.indices.shape[0]


def test_kbet_per_label():
    X, labels, batch = dummy_x_labels_batch(x_is_neighbors_results=True)
    score = scib_metrics.kbet_per_label(X, batch, labels)
    assert isinstance(score, float)


def test_graph_connectivity():
    X, labels = dummy_x_labels(symmetric_positive=True, x_is_neighbors_results=True)
    metric = scib_metrics.graph_connectivity(X, labels)
    assert isinstance(metric, float)
