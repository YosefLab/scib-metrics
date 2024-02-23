import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse.csgraph import connected_components

from scib_metrics.nearest_neighbors import NeighborsResults


def graph_connectivity(X: NeighborsResults, labels: npt.NDArray) -> float:
    """Quantify the connectivity of the subgraph per cell type label.

    Parameters
    ----------
    X
        A :class:`scib_metrics.nearest_neighbors.NeighborsResults` object containing information
        about each cell's K nearest neighbors.
    labels
        Array of shape `(n_cells,)` representing label values for each cell.

    Returns
    -------
    Mean connectivity of the subgraph per cell type label.
    """
    # TODO(adamgayoso): Utils for validating inputs
    clust_res = []

    graph = X.knn_graph_distances

    for label in np.unique(labels):
        mask = labels == label
        graph_sub = graph[mask]
        graph_sub = graph_sub[:, mask]
        _, comps = connected_components(graph_sub, connection="strong")
        tab = pd.value_counts(comps)
        clust_res.append(tab.max() / sum(tab))

    return np.mean(clust_res)
