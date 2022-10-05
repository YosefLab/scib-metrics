from typing import Optional

import jax
import jax.numpy as jnp

from scib_metrics.utils import one_hot, pca

from ._types import NdArray


def pcr(
    X: NdArray,
    batch: NdArray,
    categorical: Optional[bool] = False,
    n_components: Optional[int] = None,
) -> float:
    """Principal component regression (PCR).

    Computes the overall variance contribution given a covariate according to the following formula:

    Parameters
    ----------
    X
        Array of shape (n_samples, n_features).
    batch
        Array of shape (n_samples,) or (n_samples, 1) representing batch/covariate values.
    categorical
        If True, batch will be treated as categorical and one-hot encoded.
    n_components:
        Number of components to compute, passed into :func:`~scib_metrics.utils.pca`. If None, all components are used.

    Returns
    -------
    pcr: float
        Principal component regression using the first n_components principal components.
    """
    if len(X.shape) != 2:
        raise ValueError("Dimension mismatch: X must be 2-dimensional.")
    if X.shape[0] != batch.shape[0]:
        raise ValueError("Dimension mismatch: X and batch must have the same number of samples.")

    # Batch must be 2D
    if categorical:
        batch = one_hot(jnp.resize(batch, (batch.shape[0])))
    else:
        batch = jnp.resize(batch, (batch.shape[0], 1))

    pca_results = pca(X, n_components=n_components)
    X_pca = pca_results.coordinates
    var = pca_results.variance

    # Standardize inputs - needed since no intercept in :func:`jax.numpy.linalg.lstsq`
    X_pca = (X_pca - jnp.mean(X_pca, axis=0)) / jnp.std(X_pca, axis=0)
    if not categorical:
        batch = (batch - batch.mean()) / batch.std()
    pcr = _pcr(X_pca, batch, var)
    return float(pcr)


@jax.jit
def _pcr(
    X_pca: NdArray,
    batch: NdArray,
    var: NdArray,
) -> NdArray:
    """Principal component regression.

    Parameters
    ----------
    X_pca
        Array of shape (n_samples, n_components) containing PCA coordinates. Must be standardized.
    batch
        Array of shape (n_samples, 1) or (n_samples, n_classes) containing batch/covariate values. Must be standardized
        if not categorical (one-hot).
    var
        Array of shape (n_components,) containing the explained variance of each PC.
    """

    def get_r2(pc, batch):
        rss = jnp.linalg.lstsq(batch, pc)[1]
        tss = jnp.sum((pc - jnp.mean(pc)) ** 2)
        return jnp.maximum(0, 1 - rss / tss)

    # Index PCs on axis = 1, don't index batch
    get_r2 = jax.vmap(get_r2, in_axes=(1, None))
    r2 = jnp.ravel(get_r2(X_pca, batch))

    var = var / jnp.sum(var) * 100
    r2var = jnp.sum(r2 * var) / 100

    return r2var
