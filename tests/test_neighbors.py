from scib_metrics.nearest_neighbors import jax_approx_min_k
from tests.utils.data import dummy_benchmarker_adata


def test_jax_neighbors():
    ad, emb_keys, _, _ = dummy_benchmarker_adata()
    output = jax_approx_min_k(ad.obsm[emb_keys[0]], 10)
    assert output.distances.shape == (ad.n_obs, 10)
