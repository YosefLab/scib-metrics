import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist as sp_cdist
from sklearn.metrics import silhouette_samples as sk_silhouette_samples

import scib_metrics


def test_package_has_version():
    scib_metrics.__version__


def test_cdist():
    x = jnp.array([[1, 2], [3, 4]])
    y = jnp.array([[5, 6], [7, 8]])
    assert np.allclose(scib_metrics.utils.cdist(x, y), sp_cdist(x, y))


def test_silhouette_samples():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    labels = np.array([0, 0, 1, 1])
    assert np.allclose(scib_metrics.utils.silhouette_samples(X, labels), sk_silhouette_samples(X, labels))
