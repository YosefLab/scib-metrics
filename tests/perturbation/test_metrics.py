import numpy as np
import pytest

pytest.importorskip("pertpy")

import pandas as pd

from scib_metrics.perturbation._metrics import (
    combination_additivity,
    de_rank_recovery,
    delta_correlation,
    systema_decomposition,
)
from tests.perturbation._synthetic import make_synthetic_perturbation_adata


def test_delta_correlation_perfect_prediction_scores_one():
    rng = np.random.default_rng(0)
    true_deltas = rng.normal(size=(5, 20))
    result = delta_correlation(true_deltas, true_deltas)
    np.testing.assert_allclose(result["per_perturbation"], 1.0, atol=1e-6)
    assert result["mean"] == pytest.approx(1.0, abs=1e-6)


def test_delta_correlation_anti_correlated_prediction_scores_near_minus_one():
    rng = np.random.default_rng(0)
    true_deltas = rng.normal(size=(5, 20))
    result = delta_correlation(-true_deltas, true_deltas)
    np.testing.assert_allclose(result["per_perturbation"], -1.0, atol=1e-6)


def test_delta_correlation_restricts_to_gene_indices():
    true_deltas = np.array([[1.0, 2.0, 100.0]])
    predicted_deltas = np.array([[1.0, 2.0, -100.0]])
    result = delta_correlation(predicted_deltas, true_deltas, gene_indices=[np.array([0, 1])])
    assert result["mean"] == pytest.approx(1.0, abs=1e-6)


def test_delta_correlation_raises_on_shape_mismatch():
    with pytest.raises(ValueError):
        delta_correlation(np.zeros((2, 3)), np.zeros((2, 4)))


def test_de_rank_recovery_perfect_when_top_k_matches():
    predicted_deltas = np.array([[5.0, 0.1, -4.0, 0.2, 0.3]])
    true_de_gene_indices = [np.array([0, 2])]
    result = de_rank_recovery(predicted_deltas, true_de_gene_indices, k=2)
    assert result["per_perturbation"][0] == pytest.approx(1.0)
    assert result["mean"] == pytest.approx(1.0)


def test_de_rank_recovery_zero_when_top_k_disjoint():
    predicted_deltas = np.array([[5.0, 0.1, -4.0, 0.2, 0.3]])
    true_de_gene_indices = [np.array([1, 3])]
    result = de_rank_recovery(predicted_deltas, true_de_gene_indices, k=2)
    assert result["per_perturbation"][0] == pytest.approx(0.0)


def test_de_rank_recovery_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        de_rank_recovery(np.zeros((2, 5)), [np.array([0])], k=1)


def test_systema_decomposition_perfect_prediction_scores_one_on_both():
    rng = np.random.default_rng(0)
    true_deltas = rng.normal(size=(4, 20))
    result = systema_decomposition(true_deltas, true_deltas)
    assert result["shared"] == pytest.approx(1.0, abs=1e-6)
    assert result["specific"] == pytest.approx(1.0, abs=1e-6)


def test_systema_decomposition_shared_only_prediction_scores_low_on_specific():
    rng = np.random.default_rng(0)
    true_deltas = rng.normal(size=(4, 20))
    shared_only_prediction = np.tile(true_deltas.mean(axis=0), (4, 1))
    result = systema_decomposition(shared_only_prediction, true_deltas)
    assert result["shared"] == pytest.approx(1.0, abs=1e-6)
    assert result["specific"] < 0.5


def test_systema_decomposition_raises_with_fewer_than_two_perturbations():
    with pytest.raises(ValueError):
        systema_decomposition(np.zeros((1, 5)), np.zeros((1, 5)))


def test_combination_additivity_scores_perfect_additivity():
    adata = make_synthetic_perturbation_adata()
    pt = pytest.importorskip("pertpy")
    pseudobulk = pt.tl.PseudobulkSpace().compute(adata, target_col="perturbation", mode="mean")
    result = combination_additivity(pseudobulk, target_col="perturbation", reference_key="control")
    assert isinstance(result, pd.DataFrame)
    assert result.loc["A+B", "distance"] == pytest.approx(0.0, abs=0.1)


def test_combination_additivity_returns_empty_when_no_combinations_present():
    pt = pytest.importorskip("pertpy")
    adata = make_synthetic_perturbation_adata()
    singles_only = adata[adata.obs["perturbation"].isin(["control", "A", "B"])].copy()
    pseudobulk = pt.tl.PseudobulkSpace().compute(singles_only, target_col="perturbation", mode="mean")
    result = combination_additivity(pseudobulk, target_col="perturbation", reference_key="control")
    assert result.empty
    assert list(result.columns) == ["distance", "predicted_magnitude", "measured_magnitude"]
