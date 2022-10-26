from itertools import product

import pytest

import scib_metrics
from tests.utils.sampling import categorical_sample, normal_sample, poisson_sample

PCR_COMPARISON_PARAMS = list(product([100], [100], [False, True]))


@pytest.mark.parametrize("n_obs, n_vars, categorical", PCR_COMPARISON_PARAMS)
def test_pcr_comparison(n_obs, n_vars, categorical):
    X_pre = poisson_sample(n_obs, n_vars, seed=0)
    X_post = poisson_sample(n_obs, n_vars, seed=1)
    covariate = categorical_sample(n_obs, int(n_obs / 5)) if categorical else normal_sample(n_obs, seed=0)

    score = scib_metrics.pcr_comparison(X_pre, X_post, covariate, scale=True)
    assert score >= 0 and score <= 1
