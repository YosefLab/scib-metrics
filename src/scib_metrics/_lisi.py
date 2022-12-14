import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from scib_metrics.utils import compute_simpson_index, convert_knn_graph_to_idx


def lisi_knn(X: csr_matrix, labels: np.ndarray, perplexity: float = None) -> np.ndarray:
    """Compute the local inverse simpson index (LISI) for each cell :cite:p:`korsunsky2019harmony`.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_cells) with non-zero values
        representing distances to exactly each cell's k nearest neighbors.
    labels
        Array of shape (n_cells,) representing label values
        for each cell.
    perplexity
        Parameter controlling effective neighborhood size. If None, the
        perplexity is set to the number of neighbors // 3.

    Returns
    -------
    lisi
        Array of shape (n_cells,) with the LISI score for each cell.
    """
    labels = np.asarray(pd.Categorical(labels).codes)
    knn_dists, knn_idx = convert_knn_graph_to_idx(X)

    if perplexity is None:
        perplexity = np.floor(knn_idx.shape[1] / 3)

    n_labels = len(np.unique(labels))

    simpson = compute_simpson_index(knn_dists, knn_idx, labels, n_labels, perplexity=perplexity)
    return 1 / simpson


def ilisi_knn(X: csr_matrix, batches: np.ndarray, perplexity: float = None, scale: bool = True) -> np.ndarray:
    """Compute the integration local inverse simpson index (iLISI) for each cell :cite:p:`korsunsky2019harmony`.

    Returns a scaled version of the iLISI score for each cell, by default :cite:p:`luecken2022benchmarking`.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_cells) with non-zero values
        representing distances to exactly each cell's k nearest neighbors.
    batches
        Array of shape (n_cells,) representing batch values
        for each cell.
    perplexity
        Parameter controlling effective neighborhood size. If None, the
        perplexity is set to the number of neighbors // 3.
    scale
        Scale lisi into the range [0, 1]. If True, higher values are better.

    Returns
    -------
    ilisi
        Array of shape (n_cells,) with the iLISI score for each cell.
    """
    batches = np.asarray(pd.Categorical(batches).codes)
    lisi = lisi_knn(X, batches, perplexity=perplexity)
    ilisi = np.nanmedian(lisi)
    if scale:
        nbatches = len(np.unique(batches))
        ilisi = (ilisi - 1) / (nbatches - 1)
    return ilisi


def clisi_knn(X: csr_matrix, labels: np.ndarray, perplexity: float = None, scale: bool = True) -> np.ndarray:
    """Compute the cell-type local inverse simpson index (cLISI) for each cell :cite:p:`korsunsky2019harmony`.

    Returns a scaled version of the cLISI score for each cell, by default :cite:p:`luecken2022benchmarking`.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_cells) with non-zero values
        representing distances to exactly each cell's k nearest neighbors.
    labels
        Array of shape (n_cells,) representing cell type label values
        for each cell.
    perplexity
        Parameter controlling effective neighborhood size. If None, the
        perplexity is set to the number of neighbors // 3.
    scale
        Scale lisi into the range [0, 1]. If True, higher values are better.

    Returns
    -------
    clisi
        Array of shape (n_cells,) with the cLISI score for each cell.
    """
    labels = np.asarray(pd.Categorical(labels).codes)
    lisi = lisi_knn(X, labels, perplexity=perplexity)
    clisi = np.nanmedian(lisi)
    if scale:
        nlabels = len(np.unique(labels))
        clisi = (nlabels - clisi) / (nlabels - 1)
    return clisi
