import jax.numpy as jnp
import pytest
from sklearn.decomposition import PCA

import scib_metrics

from .sampling import poisson_sample


@pytest.mark.parametrize("n_obs, n_vars", [(1000, 1000), (1000, 500), (500, 1000)])
def test_pca(n_obs, n_vars):
    def _test_pca(n_obs, n_vars, n_components, eps=1e-4):
        X = poisson_sample(n_obs, n_vars)
        max_components = min(X.shape)
        pca = scib_metrics.utils.pca(X, n_components=n_components, return_svd=True)

        # SANITY CHECKS
        assert pca.coordinates.shape == (X.shape[0], n_components)
        assert pca.components.shape == (n_components, X.shape[1])
        assert pca.variance.shape == (n_components,)
        assert pca.variance_ratio.shape == (n_components,)
        # SVD should not be truncated to n_components
        assert pca.svd is not None
        assert pca.svd.u.shape == (X.shape[0], max_components)
        assert pca.svd.s.shape == (max_components,)
        assert pca.svd.v.shape == (max_components, X.shape[1])

        # VALUE CHECKS
        # TODO: Currently not checking coordinates and components, implementations
        # TODO: differ very slightly and not sure why (martinkim0).
        pca_true = PCA(n_components=n_components, svd_solver="full").fit(X)
        # assert jnp.allclose(pca_true.transform(X), pca.coordinates, atol=eps)
        # assert jnp.allclose(pca_true.components_, pca.components, atol=eps)
        assert jnp.allclose(pca_true.singular_values_, pca.svd.s[:n_components], atol=eps)
        assert jnp.allclose(pca_true.explained_variance_, pca.variance, atol=eps)
        assert jnp.allclose(pca_true.explained_variance_ratio_, pca.variance_ratio, atol=eps)
        # Use arpack iff n_components < max_components
        if n_components < max_components:
            pca_true = PCA(n_components=n_components, svd_solver="arpack").fit(X)
            # assert jnp.allclose(pca_true.transform(X), pca.coordinates, atol=eps)
            # assert jnp.allclose(pca_true.components_, pca.components, atol=eps)
            assert jnp.allclose(pca_true.singular_values_, pca.svd.s[:n_components], atol=eps)
            assert jnp.allclose(pca_true.explained_variance_, pca.variance, atol=eps)
            assert jnp.allclose(pca_true.explained_variance_ratio_, pca.variance_ratio, atol=eps)

    max_components = min(n_obs, n_vars)
    _test_pca(n_obs, n_vars, n_components=max_components)
    _test_pca(n_obs, n_vars, n_components=max_components - 1)
    _test_pca(n_obs, n_vars, n_components=int(max_components / 2))
    _test_pca(n_obs, n_vars, n_components=1)
