import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from scib_metrics.nearest_neighbors import NeighborsResults


def graph_connectivity(X: NeighborsResults, labels: np.ndarray) -> float:
    """Quantify the connectivity of the subgraph per cell type label.

    Parameters
    ----------
    X
        A :class:`~scib_metrics.nearest_neighbors.NeighborsResults` object.
        Connected components are computed on its symmetric UMAP connectivities
        graph (:attr:`~scib_metrics.nearest_neighbors.NeighborsResults.knn_graph_connectivities`),
        matching the original scib implementation, which uses ``adata.obsp["connectivities"]``.
    labels
        Array of shape (n_cells,) representing label values
        for each cell.
    """
    # TODO(adamgayoso): Utils for validating inputs
    clust_res = []

    # Use the symmetric connectivities graph (as in scib), not the directed kNN
    # distance graph: strongly connected components on a directed kNN graph
    # split off cells that are not in the kNN of any of their neighbors.
    graph = X.knn_graph_connectivities.tocsr()

    for label in np.unique(labels):
        mask = labels == label
        if hasattr(mask, "values"):
            mask = mask.values
        graph_sub = graph[mask]
        graph_sub = graph_sub[:, mask]
        _, comps = connected_components(graph_sub, connection="strong")
        tab = pd.Series(comps).value_counts()
        clust_res.append(tab.max() / sum(tab))

    return np.mean(clust_res)
