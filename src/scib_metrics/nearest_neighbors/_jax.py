import functools

import jax
import jax.numpy as jnp
import numpy as np

from scib_metrics.utils import cdist, get_ndarray

from ._dataclass import NeighborsResults


@functools.partial(jax.jit, static_argnames=["k", "recall_target"])
def _euclidean_ann(qy: jnp.ndarray, db: jnp.ndarray, k: int, recall_target: float = 0.95):
    """Compute half squared L2 distance between query points and database points."""
    dists = cdist(qy, db)
    return jax.lax.approx_min_k(dists, k=k, recall_target=recall_target)


def jax_approx_min_k(
    X: np.ndarray, n_neighbors: int, recall_target: float = 0.95, chunk_size: int = 2048
) -> NeighborsResults:
    """Run approximate nearest neighbor search using jax.

    On TPU backends, this is approximate nearest neighbor search. On other backends, this is exact nearest neighbor search.

    Parameters
    ----------
    X
        Data matrix.
    n_neighbors
        Number of neighbors to search for.
    recall_target
        Target recall for approximate nearest neighbor search.
    chunk_size
        Number of query points to search for at once.
    """
    db = jnp.asarray(X)
    # Loop over query points in chunks
    neighbors = []
    dists = []
    for i in range(0, db.shape[0], chunk_size):
        start = i
        end = min(i + chunk_size, db.shape[0])
        qy = db[start:end]
        dist, neighbor = _euclidean_ann(qy, db, k=n_neighbors, recall_target=recall_target)
        neighbors.append(neighbor)
        dists.append(dist)
    neighbors = jnp.concatenate(neighbors, axis=0)
    dists = jnp.concatenate(dists, axis=0)
    return NeighborsResults(indices=get_ndarray(neighbors), distances=get_ndarray(dists))
