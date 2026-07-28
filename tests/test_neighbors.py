import numpy as np
import pytest
import scanpy as sc

from scib_metrics.nearest_neighbors import jax_approx_min_k, pynndescent
from tests.utils.data import dummy_benchmarker_adata


def test_jax_neighbors():
    ad, emb_keys, _, _ = dummy_benchmarker_adata()
    output = jax_approx_min_k(ad.obsm[emb_keys[0]], 10)
    assert output.distances.shape == (ad.n_obs, 10)


@pytest.mark.parametrize("n", [5, 10, 20, 21])
def test_neighbors_results(n):
    adata, embedding_keys, *_ = dummy_benchmarker_adata()
    neigh_result = pynndescent(adata.obsm[embedding_keys[0]], n_neighbors=n)
    neigh_result = neigh_result.subset_neighbors(n=n)
    new_connect = neigh_result.knn_graph_connectivities

    sc_connect = sc.neighbors._connectivity.umap(
        neigh_result.indices[:, :n], neigh_result.distances[:, :n], n_obs=adata.n_obs, n_neighbors=n
    )

    np.testing.assert_allclose(new_connect.toarray(), sc_connect.toarray())
