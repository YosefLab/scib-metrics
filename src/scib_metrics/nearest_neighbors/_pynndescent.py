import numpy as np
from pynndescent import NNDescent

from ._dataclass import NeighborsResults


def pynndescent(X: np.ndarray, n_neighbors: int, random_state: int = 0, n_jobs: int = 1) -> NeighborsResults:
    """Run pynndescent approximate nearest neighbor search.

    Parameters
    ----------
    X
        Data matrix.
    n_neighbors
        Number of neighbors to search for.
    random_state
        Random state.
    n_jobs
        Number of jobs to use.
    """
    # Variables from umap (https://github.com/lmcinnes/umap/blob/3f19ce19584de4cf99e3d0ae779ba13a57472cd9/umap/umap_.py#LL326-L327)
    # which is used by scanpy under the hood
    n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
    n_iters = max(5, int(round(np.log2(X.shape[0]))))
    max_candidates = 60

    knn_search_index = NNDescent(
        X,
        n_neighbors=n_neighbors,
        random_state=random_state,
        low_memory=True,
        n_jobs=n_jobs,
        compressed=False,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=max_candidates,
    )
    indices, distances = knn_search_index.neighbor_graph

    return NeighborsResults(indices=indices, distances=distances)
