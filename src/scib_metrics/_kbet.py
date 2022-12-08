from functools import partial
from typing import Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from scib_metrics.utils import convert_knn_graph_to_idx, get_ndarray

from ._types import NdArray


def _chi2_cdf(df: Union[int, NdArray], x: NdArray) -> float:
    """Chi2 cdf.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chdtr.html
    for explanation of gammaincc.
    """
    return jax.scipy.special.gammaincc(df / 2, x / 2)


@jax.jit
def _kbet(neigh_batch_ids: jnp.ndarray, batches: jnp.ndarray) -> float:
    k = neigh_batch_ids.shape[1]
    expected_freq = jnp.bincount(batches, length=k)
    expected_freq = expected_freq / jnp.sum(expected_freq)
    dof = len(expected_freq) - 1

    observed_counts = jax.vmap(partial(jnp.bincount, length=dof + 1))(neigh_batch_ids)
    expected_counts = expected_freq * k
    test_statistics = jnp.sum(jnp.square(observed_counts - expected_counts) / expected_counts, axis=1)
    p_values = 1 - jax.vmap(_chi2_cdf, in_axes=(None, 0))(dof, test_statistics)

    return test_statistics, p_values


def kbet(X: csr_matrix, batches: np.ndarray, alpha: float = 0.05) -> float:
    """Compute kbet :cite:p:`buttner2018`.

    This implemenation is inspired by the implementation in Pegasus:
    https://pegasus.readthedocs.io/en/stable/index.html

    A higher acceptance rate means more mixing of batches. This implemenation does
    not exactly mirror the default original implementation, as there is currently no
    `adapt` option.

    Note that this is also not equivalent to the kbet used in the original scib package,
    as that one computes kbet for each cell type label. To acheieve this, you can use this
    function in a for loop where X would be a kNN graph for each cell type label.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_cells) with non-zero values
        representing distances to exactly each cell's k nearest neighbors.
    batches
        Array of shape (n_cells,) representing batch values
        for each cell.
    alpha
        Significance level for the statistical test.

    Returns
    -------
    acceptance_rate
        Kbet acceptance rate of the sample.
    stat_mean
        Mean Kbet chi-square statistic over all cells.
    pvalue_mean
        Mean Kbet p-value over all cells.
    """
    _, knn_idx = convert_knn_graph_to_idx(X)
    # Make sure self is included
    knn_idx = np.concatenate([np.arange(knn_idx.shape[0])[:, None], knn_idx], axis=1)
    batches = np.asarray(pd.Categorical(batches).codes)
    neigh_batch_ids = batches[knn_idx]
    chex.assert_equal_shape([neigh_batch_ids, knn_idx])
    test_statistics, p_values = _kbet(neigh_batch_ids, batches)
    test_statistics = get_ndarray(test_statistics)
    p_values = get_ndarray(p_values)
    acceptance_rate = (p_values >= alpha).mean()

    return acceptance_rate, test_statistics, p_values
