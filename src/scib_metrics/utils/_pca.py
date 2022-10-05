from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from .._types import NdArray


@dataclass
class _PCAData:
    """PCA data.

    Attributes
    ----------
    coordinates
        Array of shape (n_samples, n_components) containing the PCA coordinates.
    components
        Array of shape (n_components, n_features) containing the PCA components.
    variance
        Array of shape (n_components,) containing the explained variance of each PC.
    variance_ratio
        Array of shape (n_components,) containing the explained variance ratio of each PC.
    svd
        Tuple of NdArray containing the results from truncated SVD.
    """

    coordinates: NdArray
    components: NdArray
    variance: NdArray
    variance_ratio: NdArray
    svd: Optional[Tuple[NdArray, NdArray, NdArray]] = None


def _svd_flip(
    u: NdArray,
    v: NdArray,
    u_based_decision: bool = True,
):
    """Sign correction to ensure deterministic output from SVD.

    Jax implementation of :func:`~sklearn.utils.extmath.svd_flip`.

    Parameters
    ----------
    u
        Left singular vectors of shape (M, K).
    v
        Right singular vectors of shape (K, N).
    u_based_decision
        If True, use the columns of u as the basis for sign flipping.
    """
    if u_based_decision:
        max_abs_cols = jnp.argmax(jnp.abs(u), axis=0)
        signs = jnp.sign(u[max_abs_cols, jnp.arange(u.shape[1])])
        u_ = u * signs
        v_ = v * signs[:, None]
    else:
        max_abs_rows = jnp.argmax(jnp.abs(v), axis=1)
        signs = jnp.sign(v[jnp.arange(v.shape[0]), max_abs_rows])
        u_ = u * signs
        v_ = v * signs[:, None]
    return u_, v_


def pca(
    X: NdArray,
    n_components: Optional[int] = None,
    return_svd: bool = False,
) -> _PCAData:
    """Principal component analysis (PCA).

    Parameters
    ----------
    X
        Array of shape (n_samples, n_features).
    n_components
        Number of components to keep. If None, all components are kept.
    return_svd
        If True, also return the results from SVD.

    Returns
    -------
    results: _PCAData
    """
    max_components = min(X.shape)
    if n_components and n_components > max_components:
        raise ValueError(f"n_components = {n_components} must be <= min(n_samples, n_features) = {max_components}")
    n_components = n_components or max_components

    u, s, v, variance, variance_ratio = _pca(X)

    # Select n_components
    coordinates = u[:, :n_components] * s[:n_components]
    components = v[:n_components, :]
    variance_ = variance[:n_components]
    variance_ratio_ = variance_ratio[:n_components]

    results = _PCAData(
        coordinates,
        components,
        variance_,
        variance_ratio_,
        svd=(u, s, v) if return_svd else None,
    )
    return results


@jax.jit
def _pca(
    X: NdArray,
) -> Tuple[NdArray, NdArray, NdArray, NdArray, NdArray]:
    """Principal component analysis.

    Parameters
    ----------
    X
        Array of shape (n_samples, n_features).

    Returns
    -------
    u: NdArray
        Left singular vectors of shape (M, K).
    s: NdArray
        Singular values of shape (K,).
    v: NdArray
        Right singular vectors of shape (K, N).
    variance: NdArray
        Array of shape (K,) containing the explained variance of each PC.
    variance_ratio: NdArray
        Array of shape (K,) containing the explained variance ratio of each PC.
    """
    X_ = X - jnp.mean(X, axis=0)  # center data
    u, s, v = jnp.linalg.svd(X_, full_matrices=False)  # (M, K), (K,), (K, N)
    u, v = _svd_flip(u, v)  # make deterministic

    variance = (s**2) / (X.shape[0] - 1)
    total_variance = jnp.sum(variance)
    variance_ratio = variance / total_variance

    return u, s, v, variance, variance_ratio
