import jax.numpy as jnp
import numpy as np
from scipy import stats

from scib_metrics.utils import anderson_ksamp


def test_anderson_vs_scipy():
    """Test that the Anderson-Darling test gives the same results as scipy.stats"""
    rng = np.random.default_rng()
    data = [rng.normal(size=50), rng.normal(loc=0.5, size=30)]
    orig_res = stats.anderson_ksamp(data)
    jax_res = anderson_ksamp([jnp.asarray(d) for d in data])
    assert np.isclose(orig_res.statistic, jax_res.statistic)
    assert np.allclose(orig_res.critical_values, jax_res.critical_values)
    assert np.isclose(orig_res.significance_level, jax_res.significance_level)
