import jax.numpy as jnp
from chex import dataclass
from jax import jit

from scib_metrics._types import NdArray

from ._utils import get_ndarray


@dataclass
class _SVDResult:
    """SVD result.

    Attributes
    ----------
    u
        Array of shape (n_cells, n_components) containing the left singular vectors.
    s
        Array of shape (n_components,) containing the singular values.
    v
        Array of shape (n_components, n_features) containing the right singular vectors.
    """

    u: NdArray
    s: NdArray
    v: NdArray


@dataclass
class _PCAResult:
    """PCA result.

    Attributes
    ----------
    coordinates
        Array of shape (n_cells, n_components) containing the PCA coordinates.
    components
        Array of shape (n_components, n_features) containing the PCA components.
    variance
        Array of shape (n_components,) containing the explained variance of each PC.
    variance_ratio
        Array of shape (n_components,) containing the explained variance ratio of each PC.
    svd
        Dataclass containing the SVD data.
    """

    coordinates: NdArray
    components: NdArray
    variance: NdArray
    variance_ratio: NdArray
    svd: _SVDResult | None = None


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
    else:
        max_abs_rows = jnp.argmax(jnp.abs(v), axis=1)
        signs = jnp.sign(v[jnp.arange(v.shape[0]), max_abs_rows])
    u_ = u * signs
    v_ = v * signs[:, None]
    return u_, v_


def pca(
    X: NdArray,
    n_components: int | None = None,
    return_svd: bool = False,
) -> _PCAResult:
    """Principal component analysis (PCA).

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
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
        raise ValueError(f"n_components = {n_components} must be <= min(n_cells, n_features) = {max_components}")
    n_components = n_components or max_components

    u, s, v, variance, variance_ratio = _pca(X)

    # Select n_components
    coordinates = u[:, :n_components] * s[:n_components]
    components = v[:n_components]
    variance_ = variance[:n_components]
    variance_ratio_ = variance_ratio[:n_components]

    results = _PCAResult(
        coordinates=get_ndarray(coordinates),
        components=get_ndarray(components),
        variance=get_ndarray(variance_),
        variance_ratio=get_ndarray(variance_ratio_),
        svd=_SVDResult(u=get_ndarray(u), s=get_ndarray(s), v=get_ndarray(v)) if return_svd else None,
    )
    return results


@jit
def _pca(
    X: NdArray,
) -> tuple[NdArray, NdArray, NdArray, NdArray, NdArray]:
    """Principal component analysis.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).

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
    X_ = X - jnp.mean(X, axis=0)
    u, s, v = jnp.linalg.svd(X_, full_matrices=False)
    u, v = _svd_flip(u, v)

    variance = (s**2) / (X.shape[0] - 1)
    total_variance = jnp.sum(variance)
    variance_ratio = variance / total_variance

    return u, s, v, variance, variance_ratio
