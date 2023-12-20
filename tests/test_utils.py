import time

import numpy as np
import pytest
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

from scib_metrics.utils._utils import compute_connectivities_umap
from tests.utils.data import dummy_benchmarker_adata


@pytest.mark.parametrize("n", [5, 10, 20, 21])
def test_compute_connectivities_umap(n):
    adata, embedding_keys, *_ = dummy_benchmarker_adata()
    neigh = NearestNeighbors(n_neighbors=25).fit(adata.obsm[embedding_keys[0]])
    dist, ind = neigh.kneighbors()
    new_dist, new_connect = compute_connectivities_umap(ind[:, :n], dist[:, :n], adata.n_obs, n_neighbors=n)
    sc_dist, sc_connect = sc.neighbors._compute_connectivities_umap(ind[:, :n], dist[:, :n], adata.n_obs, n_neighbors=n)
    assert (new_dist == sc_dist).todense().all()
    assert (new_connect == sc_connect).todense().all()


def test_timing_compute_connectivities_umap():
    n_obs = 10_000
    X = np.random.normal(size=(n_obs, 10))
    neigh = NearestNeighbors(n_neighbors=90).fit(X)
    dist, ind = neigh.kneighbors()

    new_start = time.perf_counter()
    compute_connectivities_umap(ind, dist, n_obs, n_neighbors=90)
    new_end = time.perf_counter()

    sc_start = time.perf_counter()
    sc.neighbors._compute_connectivities_umap(ind, dist, n_obs, n_neighbors=90)
    sc_end = time.perf_counter()

    assert new_end - new_start < sc_end - sc_start


if __name__ == "__main__":
    pytest.main([__file__])
