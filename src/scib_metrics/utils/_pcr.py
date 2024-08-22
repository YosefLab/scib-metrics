import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit

from scib_metrics._types import NdArray

from ._pca import pca
from ._utils import one_hot


def principal_component_regression(
    X: NdArray,
    covariate: NdArray,
    categorical: bool = False,
    n_components: int | None = None,
) -> float:
    """Principal component regression (PCR) :cite:p:`buttner2018`.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    covariate
        Array of shape (n_cells,) or (n_cells, 1) representing batch/covariate values.
    categorical
        If True, batch will be treated as categorical and one-hot encoded.
    n_components:
        Number of components to compute, passed into :func:`~scib_metrics.utils.pca`.
        If None, all components are used.

    Returns
    -------
    pcr: float
        Principal component regression using the first n_components principal components.
    """
    if len(X.shape) != 2:
        raise ValueError("Dimension mismatch: X must be 2-dimensional.")
    if X.shape[0] != covariate.shape[0]:
        raise ValueError("Dimension mismatch: X and batch must have the same number of samples.")
    if categorical:
        covariate = np.asarray(pd.Categorical(covariate).codes)
    else:
        covariate = np.asarray(covariate)

    covariate = one_hot(covariate) if categorical else covariate.reshape((covariate.shape[0], 1))

    pca_results = pca(X, n_components=n_components)

    # Center inputs for no intercept
    covariate = covariate - jnp.mean(covariate, axis=0)
    pcr = _pcr(pca_results.coordinates, covariate, pca_results.variance)
    return float(pcr)


@jit
def _pcr(
    X_pca: NdArray,
    covariate: NdArray,
    var: NdArray,
) -> NdArray:
    """Principal component regression.

    Parameters
    ----------
    X_pca
        Array of shape (n_cells, n_components) containing PCA coordinates. Must be standardized.
    covariate
        Array of shape (n_cells, 1) or (n_cells, n_classes) containing batch/covariate values. Must be standardized
        if not categorical (one-hot).
    var
        Array of shape (n_components,) containing the explained variance of each PC.
    """
    residual_sum = jnp.linalg.lstsq(covariate, X_pca)[1]
    total_sum = jnp.sum((X_pca - jnp.mean(X_pca, axis=0, keepdims=True)) ** 2, axis=0)
    r2 = jnp.maximum(0, 1 - residual_sum / total_sum)

    return jnp.dot(jnp.ravel(r2), var) / jnp.sum(var)
