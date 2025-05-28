import pandas as pd

from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
from scib_metrics.nearest_neighbors import jax_approx_min_k
from tests.utils.data import dummy_benchmarker_adata


def test_benchmarker():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    bm = Benchmarker(
        ad,
        batch_key,
        labels_key,
        emb_keys,
        batch_correction_metrics=BatchCorrection(),
        bio_conservation_metrics=BioConservation(),
    )
    bm.benchmark()
    results = bm.get_results()
    assert isinstance(results, pd.DataFrame)
    bm.plot_results_table()


def test_benchmarker_default():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    bm = Benchmarker(
        ad,
        batch_key,
        labels_key,
        emb_keys,
    )
    bm.benchmark()
    results = bm.get_results()
    assert isinstance(results, pd.DataFrame)
    bm.plot_results_table()


def test_benchmarker_custom_metric_booleans():
    bioc = BioConservation(
        isolated_labels=False, nmi_ari_cluster_labels_leiden=False, silhouette_label=False, clisi_knn=True
    )
    bc = BatchCorrection(kbet_per_label=False, graph_connectivity=False, ilisi_knn=True)
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    bm = Benchmarker(ad, batch_key, labels_key, emb_keys, batch_correction_metrics=bc, bio_conservation_metrics=bioc)
    bm.benchmark()
    results = bm.get_results(clean_names=False)
    assert isinstance(results, pd.DataFrame)
    assert "isolated_labels" not in results.columns
    assert "nmi_ari_cluster_labels_leiden" not in results.columns
    assert "silhouette_label" not in results.columns
    assert "clisi_knn" in results.columns
    assert "kbet_per_label" not in results.columns
    assert "graph_connectivity" not in results.columns
    assert "ilisi_knn" in results.columns


def test_benchmarker_custom_metric_callable():
    bioc = BioConservation(clisi_knn={"perplexity": 10})
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    bm = Benchmarker(
        ad, batch_key, labels_key, emb_keys, bio_conservation_metrics=bioc, batch_correction_metrics=BatchCorrection()
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False)
    assert "clisi_knn" in results.columns


def test_benchmarker_custom_near_neighs():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    bm = Benchmarker(
        ad,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
    )
    bm.prepare(neighbor_computer=jax_approx_min_k)
    bm.benchmark()
    results = bm.get_results()
    assert isinstance(results, pd.DataFrame)
    bm.plot_results_table()
