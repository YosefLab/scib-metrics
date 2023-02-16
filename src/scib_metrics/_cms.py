import warnings
from functools import partial

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from scib_metrics.utils import anderson_ksamp, convert_knn_graph_to_idx


def _cms_one_cell(
    knn_dists: jnp.ndarray, knn_cats: jnp.ndarray, n_categories: int, cell_min: int = 4, unbalanced: bool = False
):
    # filter categories with too few cells (cell_min)
    cat_counts = jnp.bincount(knn_cats, length=n_categories)
    cat_values = jnp.arange(n_categories)
    cats_to_use = jnp.where(cat_counts >= cell_min)[0]
    cat_values = cat_values[cats_to_use]
    mask = jnp.isin(knn_cats, cat_values)
    knn_cats = knn_cats[mask]
    knn_dists = knn_dists[mask]

    # do not perform AD test if only one group with enough cells is in knn.
    if len(cats_to_use) <= 1:
        p = jnp.nan if unbalanced else 0.0
    else:
        # filter cells with the same representation
        if jnp.any(knn_dists == 0):
            warnings.warn("Distances equal to 0 - cells with identical representations detected. NaN assigned!")
            p = jnp.nan
        else:
            # perform AD test with remaining cell
            res = anderson_ksamp([knn_dists[knn_cats == cat] for cat in cat_values])
            p = res.significance_level

    return p


def cell_mixing_score(X: csr_matrix, batches: np.ndarray, cell_min: int = 10, unbalanced: bool = False) -> np.ndarray:
    """Compute the cell-specific mixing score (cms) :cite:p:`lutge2021cellmixs`.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_cells) with non-zero values
        representing distances to exactly each cell's k nearest neighbors.
    labels
        Array of shape (n_cells,) representing cell type label values
        for each cell.
    cell_min
        Minimum number of cells from each group to be included into the Anderson-Darling test.
    unbalanced
        If True neighborhoods with only one batch present will be set to NaN. This way they are not included into
        any summaries or smoothing.

    Returns
    -------
    cms
        Array of shape (n_cells,) with the cms score for each cell.
    """
    categorical_type_batches = pd.Categorical(batches)
    batches = np.asarray(categorical_type_batches.codes)
    n_categories = len(categorical_type_batches.categories)
    knn_dists, knn_idx = convert_knn_graph_to_idx(X)
    knn_cats = jnp.asarray(batches[knn_idx])
    knn_dists = jnp.asarray(knn_dists)

    cms_fn = partial(_cms_one_cell, n_categories=n_categories, cell_min=cell_min, unbalanced=unbalanced)

    ps = []
    for dists, cats in zip(knn_dists, knn_cats):
        ps.append(cms_fn(dists, cats))

    # TODO: add smoothing

    return np.array(ps)
