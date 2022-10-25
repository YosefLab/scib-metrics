import warnings
from typing import Optional

import jax
import jax.numpy as jnp
from jax import jit

from scib_metrics.utils import one_hot, pca

from ._types import NdArray


def pcr_comparison(
    X_pre: NdArray,
    X_post: NdArray,
    covariate: NdArray,
    scale: bool = True,
    **kwargs,
) -> float:
    """Principal component regression (PCR) comparison :cite:p:`buttner2018`.

    Compare the explained variance before and after integration.

    Parameters
    ----------
    X_pre
        Pre-integration array of shape (n_cells, n_features).
    X_post
        Post-integration array of shape (n_celss, n_features).
    covariate_pre:
        Array of shape (n_cells,) or (n_cells, 1) representing batch/covariate values.
    scale
        Whether to scale the score between 0 and 1. If True, larger values correspond to
        larger differences in variance contributions between `X_pre` and `X_post`.
    kwargs
        Keyword arguments passed into :func:`~scib_metrics.principal_component_regression`.

    Returns
    -------
    pcr_compared: float
        Principal component regression score comparing the explained variance before and
        after integration.
    """
    if X_pre.shape[0] != X_post.shape[0]:
        raise ValueError("Dimension mismatch: `X_pre` and `X_post` must have the same number of samples.")
    if covariate.shape[0] != X_pre.shape[0]:
        raise ValueError("Dimension mismatch: `X_pre` and `covariate` must have the same number of samples.")

    pcr_pre = principal_component_regression(X_pre, covariate, **kwargs)
    pcr_post = principal_component_regression(X_post, covariate, **kwargs)

    if scale:
        pcr_compared = (pcr_pre - pcr_post) / pcr_pre
        if pcr_compared < 0:
            warnings.warn(
                "PCR comparison score is negative, meaning variance contribution "
                "increased after integration. Setting to 0."
            )
            pcr_compared = 0
    else:
        pcr_compared = pcr_post - pcr_pre

    return pcr_compared


def principal_component_regression(
    X: NdArray,
    covariate: NdArray,
    categorical: bool = False,
    n_components: Optional[int] = None,
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

    def r2(pc, batch):
        residual_sum = jnp.linalg.lstsq(batch, pc)[1]
        total_sum = jnp.sum((pc - jnp.mean(pc)) ** 2)
        return jnp.maximum(0, 1 - residual_sum / total_sum)

    # Index PCs on axis = 1, don't index batch
    r2_ = jax.vmap(r2, in_axes=(1, None))(X_pca, covariate)
    return jnp.dot(jnp.ravel(r2_), var) / jnp.sum(var)
