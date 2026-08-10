import numpy as np
import pytest

pytest.importorskip("pertpy")

from scib_metrics.perturbation._baselines import AdditiveBaseline, MeanBaseline
from tests.perturbation._synthetic import make_synthetic_perturbation_adata


def test_mean_baseline_broadcasts_same_prediction_regardless_of_identity():
    adata = make_synthetic_perturbation_adata()
    baseline = MeanBaseline().fit(adata, target_col="perturbation", reference_key="control")
    predictions = baseline.predict(["A", "B", "unseen_name"])
    assert predictions.shape == (3, 20)
    np.testing.assert_allclose(predictions[0], predictions[1])
    np.testing.assert_allclose(predictions[0], predictions[2])


def test_mean_baseline_raises_when_reference_key_missing():
    adata = make_synthetic_perturbation_adata()
    with pytest.raises(ValueError):
        MeanBaseline().fit(adata, target_col="perturbation", reference_key="not_a_real_label")


def test_additive_baseline_recovers_perfect_combination():
    adata = make_synthetic_perturbation_adata()
    train = adata[adata.obs["perturbation"].isin(["control", "A", "B"])].copy()
    baseline = AdditiveBaseline().fit(train, target_col="perturbation", reference_key="control")
    predicted = baseline.predict(["A+B"])[0]

    full_pseudobulk_a = adata[adata.obs["perturbation"] == "A"].X.mean(axis=0)
    full_pseudobulk_b = adata[adata.obs["perturbation"] == "B"].X.mean(axis=0)
    full_pseudobulk_control = adata[adata.obs["perturbation"] == "control"].X.mean(axis=0)
    true_combination_delta = (full_pseudobulk_a - full_pseudobulk_control) + (
        full_pseudobulk_b - full_pseudobulk_control
    )
    np.testing.assert_allclose(predicted, true_combination_delta, atol=1e-4)


def test_additive_baseline_falls_back_to_mean_for_non_decomposable_names():
    adata = make_synthetic_perturbation_adata()
    train = adata[adata.obs["perturbation"].isin(["control", "A", "B"])].copy()
    baseline = AdditiveBaseline().fit(train, target_col="perturbation", reference_key="control")
    predicted = baseline.predict(["totally_unseen"])[0]
    np.testing.assert_allclose(predicted, baseline.mean_delta_)
