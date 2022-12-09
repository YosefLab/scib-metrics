import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def graph_connectivity(X: csr_matrix, labels: np.ndarray) -> float:
    r"""Quantify the connectivity of the subgraph per cell type label.

    The final score is the average for all cell type labels :math:`C`, according to the equation:

    .. math::
        GC = \\frac {1} {|C|} \\sum_{c \\in C} \\frac {|{LCC(subgraph_c)}|} {|c|}

    where :math:`|LCC(subgraph_c)|` stands for all cells in the largest connected component and :math:`|c|` stands for all cells of
    cell type :math:`c`.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_cells) with non-zero values
        representing distances to exactly each cell's k nearest neighbors.
    labels
        Array of shape (n_cells,) representing label values
        for each cell.
    """
    clust_res = []

    for label in np.unique(labels):
        mask = labels == label
        graph_sub = X[mask]
        graph_sub = graph_sub[:, mask]
        _, comps = connected_components(graph_sub, connection="strong")
        tab = pd.value_counts(comps)
        clust_res.append(tab.max() / sum(tab))

    return np.mean(clust_res)
