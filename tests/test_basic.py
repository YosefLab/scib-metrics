import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist as sp_cdist
from sklearn.metrics import silhouette_samples as sk_silhouette_samples

import scib_metrics


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


def test_nmi_ari_cluster_labels():
    X, labels = dummy_x_labels()
    nmi, ari = scib_metrics.nmi_ari_cluster_labels(X, labels)
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
