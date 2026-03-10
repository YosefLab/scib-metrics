"""Tests for spatial transcriptomics metrics."""

import numpy as np
import pandas as pd
import pytest

import scib_metrics
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation, SpatialConservation
from tests.utils.data import dummy_benchmarker_adata, dummy_spatial_benchmarker_adata

# ── helpers ──────────────────────────────────────────────────────────────────


def _compact_data(seed=0):
    """4 tight spatial blobs with matching labels."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [50, 0], [0, 50], [50, 50]], dtype=float)
    labels = np.repeat(np.arange(4), 50)
    coords = centers[labels] + rng.normal(scale=2.0, size=(200, 2))
    return coords, labels


# ── MRRE ──────────────────────────────────────────────────────────────────────


def test_spatial_mrre_returns_float_in_range():
    coords, _ = _compact_data()
    rng = np.random.default_rng(3)
    emb = coords + rng.normal(scale=1.0, size=coords.shape)
    score = scib_metrics.spatial_mrre(emb, coords)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_spatial_mrre_identity_is_best():
    coords, _ = _compact_data()
    assert scib_metrics.spatial_mrre(coords, coords) > scib_metrics.spatial_mrre(
        np.random.default_rng(3).normal(size=coords.shape), coords
    )


def test_spatial_mrre_random_worse_than_correlated():
    coords, _ = _compact_data()
    rng = np.random.default_rng(4)
    corr = scib_metrics.spatial_mrre(coords + rng.normal(scale=1.0, size=coords.shape), coords)
    rand = scib_metrics.spatial_mrre(rng.normal(size=(len(coords), 10)), coords)
    assert corr > rand


# ── kNN overlap ───────────────────────────────────────────────────────────────


def test_spatial_knn_overlap_returns_float_in_range():
    coords, _ = _compact_data()
    rng = np.random.default_rng(4)
    score = scib_metrics.spatial_knn_overlap(coords + rng.normal(scale=1.0, size=coords.shape), coords)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_spatial_knn_overlap_identity_is_one():
    coords, _ = _compact_data()
    assert scib_metrics.spatial_knn_overlap(coords, coords) == pytest.approx(1.0)


def test_spatial_knn_overlap_random_lower():
    coords, _ = _compact_data()
    rng = np.random.default_rng(5)
    corr = scib_metrics.spatial_knn_overlap(coords + rng.normal(scale=0.5, size=coords.shape), coords)
    rand = scib_metrics.spatial_knn_overlap(rng.normal(size=(len(coords), 10)), coords)
    assert corr > rand


# ── Distance correlation ──────────────────────────────────────────────────────


def test_spatial_distance_correlation_returns_float_in_range():
    coords, _ = _compact_data()
    rng = np.random.default_rng(5)
    score = scib_metrics.spatial_distance_correlation(coords + rng.normal(scale=1.0, size=coords.shape), coords)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_spatial_distance_correlation_identity_is_one():
    coords, _ = _compact_data()
    assert scib_metrics.spatial_distance_correlation(coords, coords) == pytest.approx(1.0)


def test_spatial_distance_correlation_correlated_higher():
    coords, _ = _compact_data()
    rng = np.random.default_rng(5)
    corr = scib_metrics.spatial_distance_correlation(coords + rng.normal(scale=1.0, size=coords.shape), coords)
    rand = scib_metrics.spatial_distance_correlation(rng.normal(size=(len(coords), 10)), coords)
    assert corr > rand


# ── Moran's I ─────────────────────────────────────────────────────────────────


def test_spatial_morans_i_returns_float_in_range():
    coords, _ = _compact_data()
    rng = np.random.default_rng(6)
    emb = coords + rng.normal(scale=1.0, size=coords.shape)
    score = scib_metrics.spatial_morans_i(emb, coords)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_spatial_morans_i_spatially_smooth_higher():
    """A spatially smooth embedding should score higher than random noise."""
    coords, _ = _compact_data()
    rng = np.random.default_rng(7)
    smooth = coords + rng.normal(scale=0.5, size=coords.shape)
    noisy = rng.normal(size=coords.shape)
    assert scib_metrics.spatial_morans_i(smooth, coords) > scib_metrics.spatial_morans_i(noisy, coords)


# ── Benchmarker integration ───────────────────────────────────────────────────


def test_benchmarker_with_spatial_conservation():
    adata, emb_keys, batch_key, labels_key = dummy_spatial_benchmarker_adata()
    bm = Benchmarker(
        adata,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=None,
        batch_correction_metrics=None,
        spatial_conservation_metrics=SpatialConservation(),
        spatial_key="spatial",
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False)
    assert isinstance(results, pd.DataFrame)
    for col in ("spatial_mrre", "spatial_knn_overlap", "spatial_distance_correlation", "spatial_morans_i"):
        assert col in results.columns, f"Missing column: {col}"
    bm.plot_results_table()


def test_benchmarker_spatial_aggregate_present():
    """Spatial conservation aggregate column should appear in results."""
    adata, emb_keys, batch_key, labels_key = dummy_spatial_benchmarker_adata()
    bm = Benchmarker(
        adata,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=None,
        batch_correction_metrics=None,
        spatial_key="spatial",
    )
    bm.benchmark()
    results = bm.get_results()
    assert "Spatial conservation" in results.columns


def test_benchmarker_spatial_with_bio_and_batch():
    """All three metric categories run together; spatial not in Total (weight=0)."""
    adata, emb_keys, batch_key, labels_key = dummy_spatial_benchmarker_adata()
    bm = Benchmarker(
        adata,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=BioConservation(
            isolated_labels=False,
            nmi_ari_cluster_labels_leiden=False,
            nmi_ari_cluster_labels_kmeans=True,
            silhouette_label=True,
            clisi_knn=False,
        ),
        batch_correction_metrics=BatchCorrection(
            bras=True,
            ilisi_knn=False,
            kbet_per_label=False,
            graph_connectivity=False,
            pcr_comparison=False,
        ),
        spatial_key="spatial",
    )
    bm.benchmark()
    results = bm.get_results()
    assert isinstance(results, pd.DataFrame)
    assert "Total" in results.columns
    assert "Spatial conservation" in results.columns
    bm.plot_results_table()


def test_benchmarker_spatial_weight_in_total():
    """Non-zero spatial_conservation_weight should shift the Total score."""
    adata, emb_keys, batch_key, labels_key = dummy_spatial_benchmarker_adata()
    kwargs = dict(
        bio_conservation_metrics=BioConservation(
            isolated_labels=False,
            nmi_ari_cluster_labels_leiden=False,
            nmi_ari_cluster_labels_kmeans=True,
            silhouette_label=True,
            clisi_knn=False,
        ),
        batch_correction_metrics=BatchCorrection(
            bras=True,
            ilisi_knn=False,
            kbet_per_label=False,
            graph_connectivity=False,
            pcr_comparison=False,
        ),
        spatial_key="spatial",
    )
    bm0 = Benchmarker(adata, batch_key, labels_key, emb_keys, spatial_conservation_weight=0.0, **kwargs)
    bm0.benchmark()

    bm1 = Benchmarker(adata, batch_key, labels_key, emb_keys, spatial_conservation_weight=0.2, **kwargs)
    bm1.benchmark()

    # Drop the "Metric Type" row (which contains string "Aggregate score") before casting
    r0 = bm0.get_results().loc[emb_keys, "Total"].astype(float)
    r1 = bm1.get_results().loc[emb_keys, "Total"].astype(float)
    # With weight=0.2 the totals should differ from weight=0.0
    assert not np.allclose(r0.values, r1.values)


def test_benchmarker_embedding_spatial_scores_vary():
    """Embedding-based spatial metrics should differ across different embeddings."""
    adata, emb_keys, batch_key, labels_key = dummy_spatial_benchmarker_adata(n_spots=200, seed=0)
    bm = Benchmarker(
        adata,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=None,
        batch_correction_metrics=None,
        spatial_key="spatial",
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False)
    for col in ("spatial_mrre", "spatial_knn_overlap", "spatial_distance_correlation", "spatial_morans_i"):
        vals = results.loc[results.index.isin(emb_keys), col].astype(float).values
        # At least some variation expected (embeddings are independently random)
        assert not np.allclose(vals, vals[0]), f"Metric '{col}' should vary across embeddings"


def test_benchmarker_spatial_scores_in_unit_interval():
    """All spatial metric values should be in [0, 1]."""
    adata, emb_keys, batch_key, labels_key = dummy_spatial_benchmarker_adata()
    bm = Benchmarker(
        adata,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=None,
        batch_correction_metrics=None,
        spatial_key="spatial",
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False)
    for col in ("spatial_mrre", "spatial_knn_overlap", "spatial_distance_correlation", "spatial_morans_i"):
        vals = results.loc[results.index.isin(emb_keys), col].astype(float).values
        assert np.all(vals >= 0.0), f"'{col}' has values < 0"
        assert np.all(vals <= 1.0), f"'{col}' has values > 1"


# ── auto-detection / flag behaviour ──────────────────────────────────────────


def test_benchmarker_no_spatial_key_no_spatial_metrics():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    bm = Benchmarker(ad, batch_key, labels_key, emb_keys)
    bm.benchmark()
    results = bm.get_results(clean_names=False)
    for col in ("spatial_mrre", "spatial_knn_overlap", "spatial_distance_correlation", "spatial_morans_i"):
        assert col not in results.columns


def test_benchmarker_spatial_key_auto_enables_spatial_metrics():
    adata, emb_keys, batch_key, labels_key = dummy_spatial_benchmarker_adata()
    bm = Benchmarker(
        adata,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=None,
        batch_correction_metrics=None,
        spatial_key="spatial",
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False)
    for col in ("spatial_mrre", "spatial_knn_overlap", "spatial_distance_correlation", "spatial_morans_i"):
        assert col in results.columns


def test_benchmarker_spatial_key_explicit_none_disables_spatial():
    adata, emb_keys, batch_key, labels_key = dummy_spatial_benchmarker_adata()
    bm = Benchmarker(
        adata,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=None,
        batch_correction_metrics=BatchCorrection(
            bras=True,
            ilisi_knn=False,
            kbet_per_label=False,
            graph_connectivity=False,
            pcr_comparison=False,
        ),
        spatial_conservation_metrics=None,
        spatial_key="spatial",
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False)
    for col in ("spatial_mrre", "spatial_knn_overlap", "spatial_distance_correlation", "spatial_morans_i"):
        assert col not in results.columns


def test_benchmarker_spatial_partial_config():
    adata, emb_keys, batch_key, labels_key = dummy_spatial_benchmarker_adata()
    bm = Benchmarker(
        adata,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=None,
        batch_correction_metrics=None,
        spatial_conservation_metrics=SpatialConservation(
            spatial_knn_overlap=False,
            spatial_distance_correlation=False,
        ),
        spatial_key="spatial",
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False)
    assert "spatial_mrre" in results.columns
    assert "spatial_morans_i" in results.columns
    assert "spatial_knn_overlap" not in results.columns
    assert "spatial_distance_correlation" not in results.columns


def test_benchmarker_spatial_missing_key_raises():
    adata, emb_keys, batch_key, labels_key = dummy_spatial_benchmarker_adata()
    with pytest.raises(ValueError, match="spatial_key must be provided"):
        Benchmarker(
            adata,
            batch_key,
            labels_key,
            emb_keys,
            bio_conservation_metrics=None,
            batch_correction_metrics=None,
            spatial_conservation_metrics=SpatialConservation(),
            spatial_key=None,
        )


def test_benchmarker_spatial_wrong_key_raises():
    adata, emb_keys, batch_key, labels_key = dummy_spatial_benchmarker_adata()
    with pytest.raises(ValueError, match="not found in adata.obsm"):
        Benchmarker(
            adata,
            batch_key,
            labels_key,
            emb_keys,
            spatial_conservation_metrics=SpatialConservation(),
            spatial_key="nonexistent_key",
        )
