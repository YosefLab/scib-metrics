import numpy as np
import pytest

pytest.importorskip("pertpy")

from scib_metrics.perturbation._baselines import MeanBaseline
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
