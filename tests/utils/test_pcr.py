from itertools import product

import numpy as np
import pandas as pd
import pytest
from scib.metrics import pc_regression

import scib_metrics
from scib_metrics.utils import get_ndarray
from tests.utils.sampling import categorical_sample, normal_sample, poisson_sample

PCR_PARAMS = list(product([10, 100, 1000], [10, 100, 1000], [False]))
# TODO(martinkim0): Currently not testing categorical covariates because of
# TODO(martinkim0): reproducibility issues with original scib. See comment in PR #16.


@pytest.mark.parametrize("n_obs, n_vars, categorical", PCR_PARAMS)
def test_pcr(n_obs, n_vars, categorical):
    def _test_pcr(n_obs: int, n_vars: int, n_components: int, categorical: bool, eps=1e-3, seed=123):
        X = poisson_sample(n_obs, n_vars, seed=seed)
        covariate = categorical_sample(n_obs, int(n_obs / 5)) if categorical else normal_sample(n_obs, seed=seed)

        pcr_true = pc_regression(
            get_ndarray(X),
            pd.Categorical(get_ndarray(covariate)) if categorical else get_ndarray(covariate),
            n_comps=n_components,
        )
        pcr = scib_metrics.utils.principal_component_regression(
            X,
            covariate,
            categorical=categorical,
            n_components=n_components,
        )
        assert np.allclose(pcr_true, pcr, atol=eps)

    max_components = min(n_obs, n_vars)
    _test_pcr(n_obs, n_vars, n_components=max_components, categorical=categorical)
    _test_pcr(n_obs, n_vars, n_components=max_components - 1, categorical=categorical)
    _test_pcr(n_obs, n_vars, n_components=int(max_components / 2), categorical=categorical)
    _test_pcr(n_obs, n_vars, n_components=1, categorical=categorical)
