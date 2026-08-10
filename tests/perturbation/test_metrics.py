import numpy as np
import pytest

pytest.importorskip("pertpy")

from scib_metrics.perturbation._metrics import delta_correlation


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
