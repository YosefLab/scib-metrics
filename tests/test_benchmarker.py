from scib_metrics.benchmark import Benchmarker
from tests.utils.data import dummy_benchmarker_adata


def test_benchmarker():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    bm = Benchmarker(ad, batch_key, labels_key, emb_keys)
    bm.benchmark()
