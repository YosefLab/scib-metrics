from typing import Optional

import jax
import jax.numpy as jnp

from ._types import NdArray
from scib_metrics.utils import pca, one_hot


def pc_regression(
    X: NdArray, 
    batch: NdArray,
    categorical: Optional[bool] = False,
    n_components: Optional[int] = None,
) -> float:
    """Principal component regression (PCR) [Buttner18_].

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

    pcr = _pcr(X_pca, batch, var)
    return pcr


@jax.jit
def _pcr(
    X_pca: NdArray,
    batch: NdArray,
    var: NdArray,
) -> float:
    def get_r2(pc, batch):
        rss = jnp.linalg.lstsq(batch, pc)[1]
        tss = jnp.sum((pc - jnp.mean(pc))**2)
        return 1 - rss / tss
        
    # Index PCs on axis = 1, don't index batch
    get_r2 = jax.vmap(get_r2, in_axes=(1, None)) 
    r2 = jnp.ravel(get_r2(X_pca, batch))

    var = var / jnp.sum(var) * 100
    r2var = jnp.sum(r2 * var) / 100

    return r2var
