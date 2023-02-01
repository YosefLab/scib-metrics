import functools

import jax
import jax.numpy as jnp
import numpy as np

from ._dataclass import NeighborsOutput


@functools.partial(jax.jit, static_argnames=["k", "recall_target"])
def _l2_ann(qy, db, half_db_norms, k=10, recall_target=0.95):
    dists = half_db_norms - jax.lax.dot(qy, db.transpose())
    return jax.lax.approx_min_k(dists, k=k, recall_target=recall_target)


def _get_ndarray(x):
    if isinstance(x, jnp.ndarray):
        return np.asarray(jax.device_get(x))
    else:
        return x


def jax_approx_min_k(
    X: np.ndarray, n_neighbors: int, recall_target: float = 0.95, chunk_size: int = 1024
) -> NeighborsOutput:
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
    db = X
    half_db_norm_sq = jnp.linalg.norm(db, axis=1) ** 2 / 2
    # Loop over query points in chunks
    neighbors = []
    dists = []
    for i in range(0, db.shape[0], chunk_size):
        qy = db[i : i + chunk_size]
        dist, neighbor = _l2_ann(qy, db, half_db_norm_sq, k=n_neighbors, recall_target=recall_target)
        neighbors.append(neighbor)
        dists.append(dist)
    neighbors = jnp.concatenate(neighbors, axis=0)
    dists = jnp.concatenate(dists, axis=0)
    return NeighborsOutput(indices=_get_ndarray(neighbors), distances=_get_ndarray(jnp.sqrt(dists)))
