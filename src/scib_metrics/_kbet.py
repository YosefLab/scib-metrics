import logging
from functools import partial
from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix

from scib_metrics.utils import convert_knn_graph_to_idx, diffusion_nn, get_ndarray

from ._types import NdArray

logger = logging.getLogger(__name__)


def _chi2_cdf(df: Union[int, NdArray], x: NdArray) -> float:
    """Chi2 cdf.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chdtr.html
    for explanation of gammainc.
    """
    return jax.scipy.special.gammainc(df / 2, x / 2)


@partial(jax.jit, static_argnums=2)
def _kbet(neigh_batch_ids: jnp.ndarray, batches: jnp.ndarray, n_batches: int) -> float:
    expected_freq = jnp.bincount(batches, length=n_batches)
    expected_freq = expected_freq / jnp.sum(expected_freq)
    dof = n_batches - 1

    observed_counts = jax.vmap(partial(jnp.bincount, length=n_batches))(neigh_batch_ids)
    expected_counts = expected_freq * neigh_batch_ids.shape[1]
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
    as that one computes kbet for each cell type label. To acheieve this, use
    :func:`scib_metrics.kbet_per_label`.

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

    Notes
    -----
    This function requires X to be cell-cell distances, not connectivies.
    """
    if len(batches) != X.shape[0]:
        raise ValueError("Length of batches does not match number of cells.")
    _, knn_idx = convert_knn_graph_to_idx(X)
    # Make sure self is included
    knn_idx = np.concatenate([np.arange(knn_idx.shape[0])[:, None], knn_idx], axis=1)
    batches = np.asarray(pd.Categorical(batches).codes)
    neigh_batch_ids = batches[knn_idx]
    chex.assert_equal_shape([neigh_batch_ids, knn_idx])
    n_batches = jnp.unique(batches).shape[0]
    test_statistics, p_values = _kbet(neigh_batch_ids, batches, n_batches)
    test_statistics = get_ndarray(test_statistics)
    p_values = get_ndarray(p_values)
    acceptance_rate = (p_values >= alpha).mean()

    return acceptance_rate, test_statistics, p_values


def kbet_per_label(
    X: csr_matrix,
    batches: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.05,
    diffusion_n_comps: int = 100,
    return_df: bool = False,
) -> Union[float, Tuple[float, pd.DataFrame]]:
    """Compute kBET score per cell type label as in :cite:p:`luecken2022benchmarking`.

    This approximates the method used in the original scib package. Notably, the underlying
    kbet might have some inconsistencies with the R implementation. Furthermore, to equalize
    the neighbor graphs of cell type subsets we use diffusion distance approximated with diffusion
    maps. Increasing `diffusion_n_comps` will increase the accuracy of the approximation.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_cells) with non-zero values
        representing connectivies to exactly each cell's k nearest neighbors.
    batches
        Array of shape (n_cells,) representing batch values
        for each cell.
    alpha
        Significance level for the statistical test.
    diffusion_n_comps
        Number of diffusion components to use for diffusion distance approximation.
    return_df
        Return dataframe of results in addition to score.

    Returns
    -------
    kbet_score
        Kbet score over all cells. Higher means more integrated, as in the kBET acceptance rate.
    df
        Dataframe with kBET score per cell type label.

    Notes
    -----
    This function requires X to be cell-cell connectivities, not distances.
    """
    if len(batches) != X.shape[0]:
        raise ValueError("Length of batches does not match number of cells.")
    if len(labels) != X.shape[0]:
        raise ValueError("Length of labels does not match number of cells.")
    # set upper bound for k0
    size_max = 2**31 - 1
    batches = np.asarray(pd.Categorical(batches).codes)
    labels = np.asarray(labels)

    # prepare call of kBET per cluster
    kbet_scores = {"cluster": [], "kBET": []}
    for clus in np.unique(labels):
        # subset by label
        mask = labels == clus
        X_sub = X[mask, :][:, mask]
        X_sub.sort_indices()
        n_obs = X_sub.shape[0]
        batches_sub = batches[mask]

        # check if neighborhood size too small or only one batch in subset
        if np.logical_or(n_obs < 10, len(np.unique(batches_sub)) == 1):
            logger.info(f"{clus} consists of a single batch or is too small. Skip.")
            score = np.nan
        else:
            quarter_mean = np.floor(np.mean(pd.Series(batches_sub).value_counts()) / 4).astype("int")
            k0 = np.min([70, np.max([10, quarter_mean])])
            # check k0 for reasonability
            if k0 * n_obs >= size_max:
                k0 = np.floor(size_max / n_obs).astype("int")

            n_comp, labs = scipy.sparse.csgraph.connected_components(X_sub, connection="strong")

            if n_comp == 1:  # a single component to compute kBET on
                try:
                    diffusion_n_comps = np.min([diffusion_n_comps, n_obs - 1])
                    nn_graph_sub = diffusion_nn(X_sub, k=k0, n_comps=diffusion_n_comps).astype("float")
                    # call kBET
                    score, _, _ = kbet(
                        nn_graph_sub,
                        batches=batches_sub,
                        alpha=alpha,
                    )
                except ValueError:
                    logger.info("Diffusion distance failed. Skip.")
                    score = 0  # i.e. 100% rejection

            else:
                # check the number of components where kBET can be computed upon
                comp_size = pd.value_counts(labs)
                # check which components are small
                comp_size_thresh = 3 * k0
                idx_nonan = np.flatnonzero(np.in1d(labs, comp_size[comp_size >= comp_size_thresh].index))

                # check if 75% of all cells can be used for kBET run
                if len(idx_nonan) / len(labs) >= 0.75:
                    # create another subset of components, assume they are not visited in a diffusion process
                    X_sub_sub = X_sub[idx_nonan, :][:, idx_nonan]
                    X_sub_sub.sort_indices()

                    try:
                        diffusion_n_comps = np.min([diffusion_n_comps, X_sub_sub.shape[0] - 1])
                        nn_graph_sub_sub = diffusion_nn(X_sub_sub, k=k0, n_comps=diffusion_n_comps).astype("float")
                        # call kBET
                        score, _, _ = kbet(
                            nn_graph_sub_sub,
                            batches=batches_sub[idx_nonan],
                            alpha=alpha,
                        )
                    except ValueError:
                        logger.info("Diffusion distance failed. Skip.")
                        score = 0  # i.e. 100% rejection
                else:  # if there are too many too small connected components, set kBET score to 0
                    score = 0  # i.e. 100% rejection

        kbet_scores["cluster"].append(clus)
        kbet_scores["kBET"].append(score)

    kbet_scores = pd.DataFrame.from_dict(kbet_scores)
    kbet_scores = kbet_scores.reset_index(drop=True)

    final_score = np.nanmean(kbet_scores["kBET"])
    if not return_df:
        return final_score
    else:
        return final_score, kbet_scores
