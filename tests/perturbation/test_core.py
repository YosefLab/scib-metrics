import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pertpy")

from scib_metrics.perturbation._core import PerturbationBaselines, PerturbationBenchmarker, PerturbationMetrics
from tests.perturbation._synthetic import make_synthetic_perturbation_adata


def _make_train_test_split():
    adata = make_synthetic_perturbation_adata()
    train = adata[adata.obs["perturbation"].isin(["control", "A", "B"])].copy()
    test = adata[adata.obs["perturbation"] == "A+B"].copy()
    return train, test


def _perturbation_encodings():
    # `LinearBaseline` (Task 5) hard-requires a `perturbation_encodings` entry for every
    # trained and held-out name whenever it's enabled (`PerturbationBaselines.linear=True`
    # by default) -- see its docstring ("Required if `baselines.linear` is enabled."). A
    # multi-hot encoding over the base perturbations ("A", "B") that composes additively for
    # combinations ("A+B" = "A" | "B") is the natural minimal encoding for this synthetic
    # fixture, and lets `LinearBaseline` generalize to the held-out combination.
    return {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0]), "A+B": np.array([1.0, 1.0])}


def test_benchmarker_runs_default_baselines_and_metrics():
    train, test = _make_train_test_split()
    benchmarker = PerturbationBenchmarker(train, test, perturbation_encodings=_perturbation_encodings())
    benchmarker.benchmark()
    results = benchmarker.get_results(min_max_scale=False)
    assert isinstance(results, pd.DataFrame)
    assert set(results.index) == {"mean", "additive", "linear"}
    assert results.loc["mean", "is_baseline"]
    assert "delta_correlation" in results.columns
    assert "systema_shared" not in results.columns  # only 1 held-out perturbation: A+B


def test_benchmarker_includes_user_predictions_alongside_baselines():
    train, test = _make_train_test_split()

    user_predictions = {"my_model": np.zeros((1, 20))}
    benchmarker = PerturbationBenchmarker(
        train, test, predictions=user_predictions, perturbation_encodings=_perturbation_encodings()
    )
    benchmarker.benchmark()
    results = benchmarker.get_results(min_max_scale=False)
    assert "my_model" in results.index
    assert not results.loc["my_model", "is_baseline"]


def test_benchmarker_can_disable_a_baseline():
    train, test = _make_train_test_split()
    benchmarker = PerturbationBenchmarker(train, test, baselines=PerturbationBaselines(linear=False))
    benchmarker.benchmark()
    results = benchmarker.get_results(min_max_scale=False)
    assert "linear" not in results.index


def test_benchmarker_raises_before_get_results_without_benchmark():
    train, test = _make_train_test_split()
    benchmarker = PerturbationBenchmarker(train, test)
    with pytest.raises(RuntimeError):
        benchmarker.get_results()


def test_benchmarker_de_rank_recovery_requires_true_de_gene_indices():
    train, test = _make_train_test_split()
    benchmarker = PerturbationBenchmarker(
        train,
        test,
        metrics=PerturbationMetrics(de_rank_recovery=True),
        perturbation_encodings=_perturbation_encodings(),
    )
    with pytest.raises(ValueError, match="true_de_gene_indices"):
        benchmarker.benchmark()
