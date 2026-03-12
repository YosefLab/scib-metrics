"""Tests for spatial transcriptomics metrics."""

import numpy as np
import pandas as pd
import pytest

import scib_metrics
from scib_metrics.benchmark import (
    BatchCorrection,
    Benchmarker,
    BioConservation,
    CoordinatePreservation,
    DomainBoundary,
    NichePreservation,
    SpatialConservation,
)
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
    """Coordinate preservation aggregate column should appear in results."""
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
    assert "Coordinate preservation" in results.columns


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
    assert "Coordinate preservation" in results.columns
    bm.plot_results_table()


def test_benchmarker_spatial_weight_in_total():
    """Non-zero spatial_conservation_weight should shift the Total score."""
    adata, emb_keys, batch_key, labels_key = dummy_spatial_benchmarker_adata()
    bio = BioConservation(
        isolated_labels=False,
        nmi_ari_cluster_labels_leiden=False,
        nmi_ari_cluster_labels_kmeans=True,
        silhouette_label=True,
        clisi_knn=False,
    )
    batch = BatchCorrection(
        bras=True,
        ilisi_knn=False,
        kbet_per_label=False,
        graph_connectivity=False,
        pcr_comparison=False,
    )
    bm0 = Benchmarker(
        adata,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=bio,
        batch_correction_metrics=batch,
        spatial_key="spatial",
        spatial_conservation_weight=0.0,
    )
    bm0.benchmark()

    bm1 = Benchmarker(
        adata,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=bio,
        batch_correction_metrics=batch,
        spatial_key="spatial",
        spatial_conservation_weight=0.2,
    )
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


# ── Metric differentiation: each metric uniquely sensitive to its distortion ──
#
# All four spatial metrics measure "does the embedding preserve spatial structure?"
# but from fundamentally different angles:
#
#   • distance_correlation — ALL pairwise distance ranks (global structure).
#   • kNN overlap          — SET MEMBERSHIP of the local k-NN (local structure).
#   • MRRE                 — RANK ORDER within the local k-NN (stricter than kNN).
#   • Moran's I            — SPATIAL AUTOCORRELATION of embedding values;
#                            does NOT compare distances — asks whether adjacent
#                            cells have similar embeddings regardless of scale.
#
# In real notebooks all four tend to agree because a model either captures
# spatial structure (all high) or doesn't (all moderate).  The tests below
# use four synthetic distortion types that isolate each metric's blind spot,
# demonstrating when they MUST disagree.


def _four_cluster_coords(n_per: int = 60, sep: float = 50.0, spread: float = 1.0, seed: int = 42):
    """4 tight, well-separated 2-D clusters (sep >> spread)."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [sep, 0], [0, sep], [sep, sep]], dtype=float)
    labels = np.repeat(np.arange(4), n_per)
    coords = centers[labels] + rng.normal(scale=spread, size=(4 * n_per, 2))
    return coords, labels


def test_distance_correlation_uniquely_captures_global_structure():
    """
    'Global-only' embedding: cluster centroids are preserved in the embedding
    but within-cluster positions are replaced by large independent noise.

    Between-cluster pairs dominate the pairwise distance matrix (75 % of pairs)
    and are correctly ordered in both spaces  → distance_correlation HIGH.
    The k spatial neighbours of every cell are specific nearby cells within the
    same cluster; the random intra-cluster noise picks different cells as
    latent neighbours  → kNN overlap and MRRE clearly LOWER.

    This shows that distance_correlation is uniquely sensitive to global
    distance structure and can be high even when local neighbourhoods are wrong.
    """
    coords, labels = _four_cluster_coords()
    rng = np.random.default_rng(1)
    centers = np.array([coords[labels == l].mean(0) for l in range(4)])
    # Centroid preserved; within-cluster noise >> spread but << sep
    emb = centers[labels] + rng.normal(scale=8.0, size=coords.shape)

    dist_corr = scib_metrics.spatial_distance_correlation(emb, coords)
    knn = scib_metrics.spatial_knn_overlap(emb, coords)
    mrre = scib_metrics.spatial_mrre(emb, coords)

    assert dist_corr > 0.80, f"distance_correlation should be high: {dist_corr:.3f}"
    assert knn < 0.40, f"knn_overlap should be low: {knn:.3f}"
    assert mrre < 0.65, f"mrre should be low: {mrre:.3f}"
    assert dist_corr > knn + 0.50, (
        f"distance_correlation ({dist_corr:.3f}) should clearly exceed "
        f"knn_overlap ({knn:.3f}) for a global-only embedding"
    )


def test_knn_overlap_uniquely_captures_local_neighbourhood():
    """
    'Local-only' embedding: within-cluster relative positions are preserved
    exactly, but each cluster is translated by a large random offset, completely
    destroying between-cluster distances.

    The k spatial neighbours of every cell lie within its cluster; the embedding
    shifts the whole cluster as a rigid body, so the same k cells remain the k
    nearest in embedding space  → kNN overlap = 1.0, MRRE = 1.0.
    Between-cluster distances are random (offsets >> sep)
    → distance_correlation clearly LOWER.

    This shows that kNN overlap and MRRE are uniquely sensitive to local
    neighbourhood membership and can be perfect even when global structure fails.
    """
    coords, labels = _four_cluster_coords()
    rng = np.random.default_rng(2)
    centers = np.array([coords[labels == l].mean(0) for l in range(4)])
    # Subtract cluster centre (preserves relative positions), add random large offset
    offsets = rng.normal(scale=500.0, size=(4, 2))
    emb = (coords - centers[labels]) + offsets[labels]

    knn = scib_metrics.spatial_knn_overlap(emb, coords)
    mrre = scib_metrics.spatial_mrre(emb, coords)
    dist_corr = scib_metrics.spatial_distance_correlation(emb, coords)

    assert knn > 0.90, f"knn_overlap should be near 1: {knn:.3f}"
    assert mrre > 0.90, f"mrre should be near 1: {mrre:.3f}"
    assert knn > dist_corr + 0.10, (
        f"knn_overlap ({knn:.3f}) should clearly exceed "
        f"distance_correlation ({dist_corr:.3f}) for a local-only embedding"
    )


def test_morans_i_uniquely_captures_spatial_smoothness():
    """
    'Cluster-smooth' embedding: every cell is mapped to a randomly chosen
    cluster centroid (not its own cluster's position) plus tiny noise.

    Spatially adjacent cells all belong to the same physical cluster, so they
    all receive the same (random) centroid embedding  → embedding values are
    perfectly constant within each neighbourhood  → Moran's I = 1.0.

    The k latent neighbours of a cell are determined by tiny independent noise
    rather than physical proximity  → kNN overlap and MRRE clearly LOWER.

    This shows that Moran's I is uniquely sensitive to spatial smoothness of
    the embedding signal, independently of whether exact neighbours are correct.
    """
    coords, labels = _four_cluster_coords()
    rng = np.random.default_rng(99)
    # Each cluster maps to a random point in embedding space
    random_centroids = rng.normal(scale=50.0, size=(4, 2))
    emb = random_centroids[labels] + rng.normal(scale=0.01, size=coords.shape)

    morans = scib_metrics.spatial_morans_i(emb, coords)
    knn = scib_metrics.spatial_knn_overlap(emb, coords)
    mrre = scib_metrics.spatial_mrre(emb, coords)

    assert morans > 0.95, f"spatial_morans_i should be near 1: {morans:.3f}"
    assert knn < 0.40, f"knn_overlap should be low: {knn:.3f}"
    assert mrre < 0.65, f"mrre should be low: {mrre:.3f}"
    assert morans > knn + 0.50, (
        f"spatial_morans_i ({morans:.3f}) should clearly exceed knn_overlap ({knn:.3f}) for a cluster-smooth embedding"
    )


def test_mrre_uniquely_sensitive_to_rank_order_within_knn():
    """
    'Rank-reversed' embedding: for a 1-D spatial layout the k-NN SET is
    identical in spatial and embedding spaces, but the rank ORDER within
    that set is reversed.

    Construction: cells are placed on a 1-D line (y = 0).  The embedding is
    2-D with emb_y alternating 0 / delta (delta = 3 > sqrt(3)).  For every
    even cell i, distance-2 neighbours (even, same emb_y) appear CLOSER in
    the embedding than distance-1 neighbours (odd, large emb_y difference),
    inverting the rank order while keeping the same 4-member k-NN set.

    Expected:
      kNN overlap ≈ 1.0  (same 4 members as spatial k-NN)
      MRRE ≈ 0.5         (rank order within k-NN inverted)

    This shows MRRE is strictly more demanding than kNN overlap: it penalises
    incorrect ordering even when set membership is perfect.
    """
    n = 80
    k = 4
    delta = 3.0  # > sqrt(3) ≈ 1.73, ensuring distance-2 < distance-1 in embedding
    coords_1d = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    emb_y = np.where(np.arange(n) % 2 == 0, 0.0, delta)
    emb = np.column_stack([np.arange(n, dtype=float), emb_y])

    knn = scib_metrics.spatial_knn_overlap(emb, coords_1d, k=k)
    mrre = scib_metrics.spatial_mrre(emb, coords_1d, k=k)

    assert knn > 0.90, f"knn_overlap should be near 1 (same members): {knn:.3f}"
    assert mrre < 0.65, f"mrre should be low (reversed ranks): {mrre:.3f}"
    assert knn > mrre + 0.30, (
        f"knn_overlap ({knn:.3f}) should clearly exceed mrre ({mrre:.3f}) when rank order within k-NN is reversed"
    )


# ── PAS ───────────────────────────────────────────────────────────────────────


def test_spatial_pas_returns_float_in_range():
    coords, _ = _compact_data()
    rng = np.random.default_rng(10)
    emb = coords + rng.normal(scale=1.0, size=coords.shape)
    score = scib_metrics.spatial_pas(emb, coords, n_clusters=4)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_spatial_pas_spatially_coherent_clusters_score_higher():
    """Clusters aligned with spatial blobs should score higher than random."""
    coords, labels = _compact_data()
    rng = np.random.default_rng(11)
    # Good embedding: tight around spatial clusters → k-means recovers spatial blobs
    good_emb = coords + rng.normal(scale=0.5, size=coords.shape)
    # Bad embedding: pure noise → k-means finds random clusters
    bad_emb = rng.normal(scale=50.0, size=coords.shape)
    good = scib_metrics.spatial_pas(good_emb, coords, n_clusters=4, seed=0)
    bad = scib_metrics.spatial_pas(bad_emb, coords, n_clusters=4, seed=0)
    assert good > bad, f"good={good:.3f} should exceed bad={bad:.3f}"


# ── CHAOS ─────────────────────────────────────────────────────────────────────


def test_spatial_chaos_returns_float_in_range():
    coords, _ = _compact_data()
    rng = np.random.default_rng(12)
    emb = coords + rng.normal(scale=1.0, size=coords.shape)
    score = scib_metrics.spatial_chaos(emb, coords, n_clusters=4)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_spatial_chaos_compact_clusters_score_higher():
    """Embedding whose clusters are spatially compact should score higher."""
    coords, labels = _compact_data()
    rng = np.random.default_rng(13)
    good_emb = coords + rng.normal(scale=0.5, size=coords.shape)
    bad_emb = rng.normal(scale=50.0, size=coords.shape)
    good = scib_metrics.spatial_chaos(good_emb, coords, n_clusters=4, seed=0)
    bad = scib_metrics.spatial_chaos(bad_emb, coords, n_clusters=4, seed=0)
    assert good > bad, f"good={good:.3f} should exceed bad={bad:.3f}"


# ── Niche kNN overlap ─────────────────────────────────────────────────────────


def test_spatial_niche_knn_overlap_returns_float_in_range():
    coords, _ = _compact_data()
    rng = np.random.default_rng(14)
    emb = coords + rng.normal(scale=1.0, size=coords.shape)
    X_expr = coords + rng.normal(scale=0.5, size=coords.shape)
    score = scib_metrics.spatial_niche_knn_overlap(emb, coords, X_expression=X_expr)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_spatial_niche_knn_overlap_fallback_no_expression():
    """Should work without X_expression (falls back to spatial coord averaging)."""
    coords, _ = _compact_data()
    rng = np.random.default_rng(15)
    emb = coords + rng.normal(scale=1.0, size=coords.shape)
    score = scib_metrics.spatial_niche_knn_overlap(emb, coords, X_expression=None)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_spatial_niche_knn_overlap_niche_aware_higher():
    """Embedding that captures niche structure should outscore random noise."""
    coords, labels = _compact_data()
    rng = np.random.default_rng(16)
    X_expr = coords + rng.normal(scale=0.5, size=coords.shape)  # expr correlated with space
    good_emb = coords + rng.normal(scale=0.5, size=coords.shape)
    rand_emb = rng.normal(size=(len(coords), 10))
    good = scib_metrics.spatial_niche_knn_overlap(good_emb, coords, X_expression=X_expr)
    rand = scib_metrics.spatial_niche_knn_overlap(rand_emb, coords, X_expression=X_expr)
    assert good > rand, f"good={good:.3f} should exceed rand={rand:.3f}"


# ── New dataclasses / Benchmarker integration ─────────────────────────────────


def test_niche_preservation_benchmarker():
    """NichePreservation runs through the Benchmarker pipeline."""
    adata, emb_keys, batch_key, labels_key = dummy_spatial_benchmarker_adata()
    bm = Benchmarker(
        adata,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=None,
        batch_correction_metrics=None,
        spatial_conservation_metrics=None,
        niche_preservation=NichePreservation(),
        spatial_key="spatial",
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False)
    assert "spatial_niche_knn_overlap" in results.columns
    vals = results.loc[results.index.isin(emb_keys), "spatial_niche_knn_overlap"].astype(float).values
    assert np.all(vals >= 0.0) and np.all(vals <= 1.0)


def test_domain_boundary_benchmarker():
    """DomainBoundary runs through the Benchmarker pipeline."""
    adata, emb_keys, batch_key, labels_key = dummy_spatial_benchmarker_adata()
    bm = Benchmarker(
        adata,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=None,
        batch_correction_metrics=None,
        spatial_conservation_metrics=None,
        domain_boundary=DomainBoundary(),
        spatial_key="spatial",
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False)
    assert "spatial_pas" in results.columns
    assert "spatial_chaos" in results.columns
    for col in ("spatial_pas", "spatial_chaos"):
        vals = results.loc[results.index.isin(emb_keys), col].astype(float).values
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0)
    bm.plot_results_table()


def test_all_three_spatial_axes_together():
    """All three spatial axes run together and produce three aggregate columns."""
    adata, emb_keys, batch_key, labels_key = dummy_spatial_benchmarker_adata()
    bm = Benchmarker(
        adata,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=None,
        batch_correction_metrics=None,
        spatial_conservation_metrics=CoordinatePreservation(),
        niche_preservation=NichePreservation(),
        domain_boundary=DomainBoundary(),
        spatial_key="spatial",
    )
    bm.benchmark()
    results = bm.get_results()
    for col in ("Coordinate preservation", "Niche preservation", "Domain boundary"):
        assert col in results.columns, f"Missing aggregate column: {col}"
    bm.plot_results_table()


def test_spatial_conservation_alias_still_works():
    """SpatialConservation is still a valid alias for CoordinatePreservation."""
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
    results = bm.get_results()
    assert "Coordinate preservation" in results.columns


def test_niche_domain_missing_spatial_key_raises():
    """niche_preservation and domain_boundary require spatial_key."""
    adata, emb_keys, batch_key, labels_key = dummy_spatial_benchmarker_adata()
    with pytest.raises(ValueError, match="spatial_key must be provided"):
        Benchmarker(
            adata,
            batch_key,
            labels_key,
            emb_keys,
            bio_conservation_metrics=None,
            batch_correction_metrics=None,
            spatial_conservation_metrics=None,
            niche_preservation=NichePreservation(),
            spatial_key=None,
        )
    with pytest.raises(ValueError, match="spatial_key must be provided"):
        Benchmarker(
            adata,
            batch_key,
            labels_key,
            emb_keys,
            bio_conservation_metrics=None,
            batch_correction_metrics=None,
            spatial_conservation_metrics=None,
            domain_boundary=DomainBoundary(),
            spatial_key=None,
        )
