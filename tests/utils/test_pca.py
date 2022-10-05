import jax.numpy as jnp
from sklearn.decomposition import PCA

import scib_metrics

from .sampling import poisson_sample


def test_pca():
    def _test_pca(n_obs, n_vars, n_components):
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
        sklearn_pca = PCA(n_components=n_components, svd_solver="full")
        pca_true = sklearn_pca.fit(X)
        assert jnp.allclose(sklearn_pca.fit_transform(X), pca.coordinates)
        assert jnp.allclose(pca_true.components_, pca.components)
        assert jnp.allclose(pca_true.explained_variance_, pca.variance)
        assert jnp.allclose(pca_true.explained_variance_ratio_, pca.variance_ratio)
        # Use arpack iff n_components < max_components
        if n_components < max_components:
            sklearn_pca = PCA(n_components=n_components, svd_solver="arpack")
            pca_true = sklearn_pca.fit(X)
            assert jnp.allclose(sklearn_pca.fit_transform(X), pca.coordinates)
            assert jnp.allclose(pca_true.components_, pca.components)
            assert jnp.allclose(pca_true.explained_variance_, pca.variance)
            assert jnp.allclose(pca_true.explained_variance_ratio_, pca.variance_ratio)

    for n_obs in [10, 100, 1000]:
        for n_vars in [10, 100, 1000]:
            max_components = min(n_obs, n_vars)
            _test_pca(n_obs, n_vars, n_components=max_components)
            _test_pca(n_obs, n_vars, n_components=int(max_components / 2))
            _test_pca(n_obs, n_vars, n_components=1)
